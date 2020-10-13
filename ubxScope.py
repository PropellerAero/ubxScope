#!/usr/bin/env python3

import ubx
from ubx import parseUBXMessage, UBXManager
from pathlib import Path
from queue import Queue
from functools import partial

import sys
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Legend, Span, Label, Div
from bokeh.document import without_document_lock

#from scipy.interpolate import CubicSpline

TOOLTIPS = [
    ("PSD", "$y dB @ $x Hz"),
]

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

#Bin Width
#This is defined in SPAN but for convenience of plotting setup
#it is hard coded here
SPAN_BIN_COUNT=256

#Axes bounds
YMAX = 180
YMIN = 30
YMIN_LABEL = YMIN +5
YMAX_LABEL = YMAX -5

#Arbitrary initial plot frame size, for referencing annotation
#positions before the plot is scaled
PLOT_WIDTH=400
PLOT_HEIGHT=400

# GNSS Frequencies (Hz)
GPS_L1_FC = 1575420000
GPS_L2_FC = 1227600000
GPS_L5_FC = 1176450000

GLONASS_L1OF_FC = 1602000000
GLONASS_L1OF_SPACING = 562500
GLONASS_L2OF_FC = 1246000000
GLONASS_L2OF_SPACING = 437500


# Be quiet on errors.
class UBXScopeQueue(UBXManager):
  def __init__(self, ser, debug=False, eofTimeout=None, onUBXCallback=None):
        """
        :param ser: Passed to UBXManager
        :param eofTimeout: Passed to UBXManager
        """
        self._queue = Queue()
        # Reflects the has-a queue's get() and empty() methods
        self.empty = self._queue.empty
        self.onUBXCallback=onUBXCallback
        self.firstUBXMessage = True
        super(UBXScopeQueue, self).__init__(ser=ser, debug=debug, eofTimeout=eofTimeout)

  def onUBXError(self, msgClass, msgId, errMsg):
    return
  def onNMEA(self, msg):
    return
  def onNMEAError(self, msg):
    return
  def onUBX(self, msg):
    if msg.__class__.__name__ in ['SPAN', 'PVT']:
      self.onUBXCallback(msg, msg.__class__.__name__)
    #else:
      #print(f'Unhandled: {msg.__class__.__name__}')

class UBXScope:
  def __init__(self, inputBuffer):

    #Setup Plot
    self.doc = curdoc()
    self.doc.title = "UBX Scope"
    self.numRfBlocks = 2
    self.spectrumFigures = [self.numRfBlocks, None]
    self.blockMetadataLabels = [self.numRfBlocks, None]

    #Hold column layouts for each block
    self.blockColumnLayouts = [self.numRfBlocks, None]

    #Setup Data Source mapping for each block
    dataSourceDict = {}
    for block in range(self.numRfBlocks):
      dataSourceDict[f'spectrumBinCenterFreqs_{block}'] = np.zeros(SPAN_BIN_COUNT)
      dataSourceDict[f'spectrumMaxima_{block}'] = np.zeros(SPAN_BIN_COUNT)
      dataSourceDict[f'spectrumCMA_{block}'] = np.zeros(SPAN_BIN_COUNT)
      dataSourceDict[f'spectrum_{block}'] = np.zeros(SPAN_BIN_COUNT)
    self.spectrumDataSource=ColumnDataSource(data=dataSourceDict)

    #Add a figure for each block
    for block in range(self.numRfBlocks):

      figure_ = figure(title=f"UBX SPAN Block {block+1}",
                      output_backend="webgl",
                      y_range=(YMIN,YMAX),
                      tooltips=TOOLTIPS,
                      tools=TOOLS,
                      plot_width=PLOT_WIDTH,
                      plot_height=PLOT_HEIGHT)

      # Add instantaneous, avg, and max line plots
      spectrum = figure_.line(source=self.spectrumDataSource,
                   x=f'spectrumBinCenterFreqs_{block}',
                   y=f'spectrum_{block}',
                   line_width=1,
                   line_color='blue')
      spectrumMax = figure_.line(source=self.spectrumDataSource,
                   x=f'spectrumBinCenterFreqs_{block}',
                   y=f'spectrumMaxima_{block}',
                   line_width=1,
                   line_color='red')
      spectrumCMA = figure_.line(source=self.spectrumDataSource,
                   x=f'spectrumBinCenterFreqs_{block}',
                   y=f'spectrumCMA_{block}',
                   line_width=1,
                   line_color='green')

      #Use an event to update the position of some labels
      #figure_.on_change("inner_width", self.figureOnChangeHandler)
      #figure_.on_change("inner_height", self.figureOnChangeHandler)

      #Label Axes
      figure_.xaxis.axis_label = "Frequency (Hz)"
      figure_.yaxis.axis_label = "Received Power dB (Unref)"

      #Legend
      legend = Legend(items=[
        ("PSD"   , [spectrum]),
        ("Max PSD" , [spectrumMax]),
        ("Avg PSD" , [spectrumCMA]),
      ], location="center",
        click_policy="hide")
      figure_.add_layout(legend, 'left')

      self.spectrumFigures[block] = figure_

      #Centre Frequencies
      gpsL1fC = Span(location=GPS_L1_FC,dimension='height', line_color='orange',line_dash='dashed', line_width=1)
      gpsL1fCLabel = Label(text='GPS L1 f₀', x=GPS_L1_FC, y=YMIN_LABEL, text_font_size='10px')
      gpsL2fC = Span(location=GPS_L2_FC,dimension='height', line_color='orange',line_dash='dashed', line_width=1)
      gpsL2fCLabel = Label(text='GPS L2 f₀', x=GPS_L2_FC, y=YMIN_LABEL, text_font_size='10px')
      gpsL5fC = Span(location=GPS_L5_FC,dimension='height', line_color='orange',line_dash='dashed', line_width=1)
      gpsL5fCLabel = Label(text='GPS L5 f₀', x=GPS_L5_FC, y=YMIN_LABEL, text_font_size='10px')

      self.spectrumFigures[block].renderers.extend([gpsL1fC, gpsL1fCLabel])
      #L2/L5 are near to each other so show on same plot, even though the hardware isn't supporting it
      self.spectrumFigures[block].renderers.extend([gpsL2fC, gpsL2fCLabel])
      self.spectrumFigures[block].renderers.extend([gpsL5fC, gpsL5fCLabel])

      #GLONASS FDMA L1OF/L2OF
      for carrier in range(-7,7):
        #L1OF
        gloL1OFf0 = GLONASS_L1OF_FC + (carrier * GLONASS_L1OF_SPACING)
        glol1OFf0Span = Span(location=gloL1OFf0, dimension='height', line_color='purple',line_dash='dashed', line_width=0.3)
        self.spectrumFigures[block].add_layout(glol1OFf0Span)

        #L2OF
        gloL2OFf0 = GLONASS_L2OF_FC + (carrier * GLONASS_L2OF_SPACING)
        gloL2OFf0Span = Span(location=gloL2OFf0, dimension='height', line_color='purple',line_dash='dashed', line_width=0.3)
        self.spectrumFigures[block].add_layout(gloL2OFf0Span)

      #Add GLONASS span labels
      gloL1OFf0Label = Label(text='GLO L1OF', x=GLONASS_L1OF_FC, y=YMIN_LABEL, text_font_size='10px')
      gloL2OFf0Label = Label(text='GLO L2OF', x=GLONASS_L2OF_FC, y=YMIN_LABEL, text_font_size='10px')
      self.spectrumFigures[block].add_layout(gloL1OFf0Label)
      self.spectrumFigures[block].add_layout(gloL2OFf0Label)

      #Metadata label
      self.blockMetadataLabels[block] = Div(text=f'NO_DATA', width=PLOT_WIDTH, height=20)

      #Create a column with rows for plot and metadata
      self.blockColumnLayouts[block] = column(row(children=[self.spectrumFigures[block]],sizing_mode="stretch_both"), self.blockMetadataLabels[block])

      #self.spectrumFigures[block].add_layout(self.blockMetadataLabels[block])


    #Row layout of columns with plot and additional metadata
    self.doc.add_root(row(children=self.blockColumnLayouts, sizing_mode="stretch_both"))

    print (f"Reading from {inputBuffer}")
    self.ubxScopeQueue = UBXScopeQueue(ser=inputBuffer, eofTimeout=0, onUBXCallback=self.onUBXMessage)
    self.ubxScopeQueue.start()

  # # Handles events generated by the plot, e.g. size change
  # def figureOnChangeHandler(self, attr, old, new):
  #   #Move labels to set a relative position
  #   #Surely there's a less awful way to do this...
  #   if attr == 'inner_height':
  #     for figure in self.spectrumFigures:
  #       print(figure.y)
  #       print(new)
  #       figure.y(1000)

  def updateSpectrumPlot(self, spectrumData, spectrumMetaData):
    #Update spectrum data
    self.spectrumDataSource.data = spectrumData

    #Update metadata
    for index, block in enumerate(spectrumMetaData):
      pgaGain = block['pga']
      self.blockMetadataLabels[index].text = f'PGA Gain: {pgaGain} dB'

  def onUBXMessage(self, msg, msgClass):

    if msgClass == 'SPAN':
      newSpectrumData = {}
      newSpectrumMetaData = [self.numRfBlocks, None]

      for block in range(msg.numRfBlocks):
        # #Interpolation is sloooowwwww
        # if False:
        #   min_x=msg.spectra[block]['spectrumBinCenterFreqs'][0]
        #   max_x=msg.spectra[block]['spectrumBinCenterFreqs'][-1]

        #   splineFn = CubicSpline(msg.spectra[block]['spectrumBinCenterFreqs'], msg.spectra[block]['spectrum'])

        #   x_interpol = np.linspace(min_x, max_x, 1000)
        #   y_interpol = splineFn(x_interpol)
        #   newSpectrumData[f'spectrumBinCenterFreqs_{block}'] = x_interpol
        #   newSpectrumData[f'spectrum_{block}'] =y_interpol

        #Centre Frequencies
        newSpectrumData[f'spectrumBinCenterFreqs_{block}'] = msg.spectra[block]['spectrumBinCenterFreqs']

        #PSD bin data
        newSpectrumData[f'spectrum_{block}'] = msg.spectra[block]['spectrum']

        #Calculate PSD max
        newSpectrumData[f'spectrumMaxima_{block}'] = np.maximum(newSpectrumData[f'spectrum_{block}'], self.spectrumDataSource.data[f'spectrumMaxima_{block}'])

        #Calculate PSD moving average
        newSpectrumData[f'spectrumCMA_{block}'] = np.mean( np.array([ newSpectrumData[f'spectrum_{block}'], self.spectrumDataSource.data[f'spectrumCMA_{block}'] ]), axis=0 )

        #Additional metadata for annotations
        newSpectrumMetaData[block] = {
          'pga': msg.spectra[block]['pga']
        }

      self.doc.add_next_tick_callback(partial(self.updateSpectrumPlot, spectrumData=newSpectrumData, spectrumMetaData=newSpectrumMetaData))

    #if msgClass == 'PVT':
      #print(msg.year)

#Read from stdin, or wherever....
inBuffer =  sys.stdin.buffer

#inFile = 'rob_roof_choke.ubx'
#ser=Path(__file__).parent.joinpath(inFile).open("rb")

scope = UBXScope(inBuffer)








