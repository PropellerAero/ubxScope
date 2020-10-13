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
LABEL_YSPACE = 2

#Arbitrary initial plot frame size, for referencing annotation
#positions before the plot is scaled
PLOT_WIDTH=400
PLOT_HEIGHT=400

#Common 'band' frequencies
X1_FC                  = 1575420000
X2_FC                  = 1227600000
X5A_FC                 = 1176450000
X5B_FC                 = 1207140000
X6_FC                  = 1278750000

#GNSS Frequencies (Hz)
#Multiple signals sharing a frequency for each system are not shown
#The signal identifier for the 'open' signal is added
#Where multiple open signals are present, no identifier is added

#GPS
GPS_L1_FC                  = X1_FC
GPS_L2_FC                  = X2_FC
GPS_L5_FC                  = X5A_FC

#Glonass FDMA
GLONASS_L1OF_FC       = 1602000000
GLONASS_L1OF_SPACING      = 562500
GLONASS_L2OF_FC       = 1246000000
GLONASS_L2OF_SPACING      = 437500

#Glonass CDMA
GLONASS_L1OC_FC       = 1600995000
GLONASS_L2OC_FC       = 1248060000
GLONASS_L3OC_FC       = 1202025000

#Galileo
GALILEO_E1_FC              = X1_FC
GALILEO_E5A_FC            = X5A_FC
GALILEO_E5B_FC            = X5B_FC
GALILEO_E5_ALTBOC_FC  = 1191795000
GALILEO_E6_FC              = X6_FC

#Beidou
BEIDOU_B1I_FC         = 1561098000
BEIDOU_B1C_FC              = X1_FC
BEIDOU_B2I_FC         = 1207140000
BEIDOU_B2A_FC             = X5A_FC
BEIDOU_B2B_FC             = X5B_FC
BEIDOU_B3I_FC         = 1268520000

#SBAS
SBAS_L1_FC                 = X1_FC
SBAS_L5_FC                = X5A_FC

#QZSS
QZSS_L1_FC                 = X1_FC
QZSS_L2_FC                 = X2_FC
QZSS_L5_FC                = X5A_FC
QZSS_L6_FC                 = X6_FC

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
      spectrumCMA.on_change('visible', self.cmaVisibleChangeHandler)
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
      #GPS
      freqAnnotationsGPS = [
        Span(location=GPS_L1_FC,dimension='height', line_color='orange',line_dash='dashed', line_width=0.5),
        Label(text='GPS L1', x=GPS_L1_FC, y=YMIN_LABEL, text_font_size='9px', text_align='center'),
        Span(location=GPS_L2_FC,dimension='height', line_color='orange',line_dash='dashed', line_width=0.5),
        Label(text='GPS L2', x=GPS_L2_FC, y=YMIN_LABEL, text_font_size='9px', text_align='center'),
        Span(location=GPS_L5_FC,dimension='height', line_color='orange',line_dash='dashed', line_width=0.5),
        Label(text='GPS L5', x=GPS_L5_FC, y=YMIN_LABEL, text_font_size='9px', text_align='center')
      ]

      freqAnnotationsGalileo = [
        Span(location=GALILEO_E1_FC,dimension='height', line_color='green',line_dash='dashed', line_width=0.5),
        Label(text='GAL E1', x=GALILEO_E1_FC, y=YMIN_LABEL+LABEL_YSPACE, text_font_size='9px', text_align='center'),
        Span(location=GALILEO_E5A_FC,dimension='height', line_color='green',line_dash='dashed', line_width=0.5),
        Label(text='GAL E5B', x=GALILEO_E5A_FC, y=YMIN_LABEL+LABEL_YSPACE, text_font_size='9px', text_align='center'),
        Span(location=GALILEO_E5B_FC,dimension='height', line_color='green',line_dash='dashed', line_width=0.5),
        Label(text='GAL E5B', x=GALILEO_E5B_FC, y=YMIN_LABEL+LABEL_YSPACE, text_font_size='9px', text_align='center'),
        Span(location=GALILEO_E5_ALTBOC_FC,dimension='height', line_color='green',line_dash='dashed', line_width=0.5),
        Label(text='GAL E5 ALTBOC', x=GALILEO_E5_ALTBOC_FC, y=YMIN_LABEL+LABEL_YSPACE, text_font_size='9px', text_align='center'),
        Span(location=GALILEO_E6_FC,dimension='height', line_color='green',line_dash='dashed', line_width=0.5),
        Label(text='GAL E6', x=GALILEO_E6_FC, y=YMIN_LABEL+LABEL_YSPACE, text_font_size='9px', text_align='center')
      ]

      #Beidou
      freqAnnotationsBeidou = [
        Span(location=BEIDOU_B1I_FC,dimension='height', line_color='red',line_dash='dashed', line_width=0.5),
        Label(text='BDS B1I', x=BEIDOU_B1I_FC, y=YMIN_LABEL+(3*LABEL_YSPACE), text_font_size='9px', text_align='center'),
        Span(location=BEIDOU_B1C_FC,dimension='height', line_color='red',line_dash='dashed', line_width=0.5),
        Label(text='BDS B1C', x=BEIDOU_B1C_FC, y=YMIN_LABEL+(3*LABEL_YSPACE), text_font_size='9px', text_align='center'),
        Span(location=BEIDOU_B2A_FC,dimension='height', line_color='red',line_dash='dashed', line_width=0.5),
        Label(text='BDS B2A', x=BEIDOU_B2A_FC, y=YMIN_LABEL+(3*LABEL_YSPACE), text_font_size='9px', text_align='center'),
        Span(location=BEIDOU_B2B_FC,dimension='height', line_color='red',line_dash='dashed', line_width=0.5),
        Label(text='BDS B2B', x=BEIDOU_B2B_FC, y=YMIN_LABEL+(3*LABEL_YSPACE), text_font_size='9px', text_align='center'),
        Span(location=BEIDOU_B3I_FC,dimension='height', line_color='red',line_dash='dashed', line_width=0.5),
        Label(text='BDS B3I', x=BEIDOU_B3I_FC, y=YMIN_LABEL+(3*LABEL_YSPACE), text_font_size='9px', text_align='center'),
      ]

      freqAnnotationsQZSS = [
        Span(location=QZSS_L1_FC,dimension='height', line_color='yellow',line_dash='dashed', line_width=0.5),
        Label(text='QZSS L1', x=QZSS_L1_FC, y=YMIN_LABEL+(4*LABEL_YSPACE), text_font_size='9px', text_align='center'),
        Span(location=QZSS_L2_FC,dimension='height', line_color='yellow',line_dash='dashed', line_width=0.5),
        Label(text='QZSS L2', x=QZSS_L2_FC, y=YMIN_LABEL+(4*LABEL_YSPACE), text_font_size='9px', text_align='center'),
        Span(location=QZSS_L5_FC,dimension='height', line_color='yellow',line_dash='dashed', line_width=0.5),
        Label(text='QZSS L5', x=QZSS_L5_FC, y=YMIN_LABEL+(4*LABEL_YSPACE), text_font_size='9px', text_align='center'),
        Span(location=QZSS_L6_FC,dimension='height', line_color='yellow',line_dash='dashed', line_width=0.5),
        Label(text='QZSS L6', x=QZSS_L6_FC, y=YMIN_LABEL+(4*LABEL_YSPACE), text_font_size='9px', text_align='center')
      ]


      freqAnnotationsGlonass = [
        Span(location=GLONASS_L1OC_FC,dimension='height', line_color='purple',line_dash='dashed', line_width=0.5),
        Label(text='GLO L1OC', x=GLONASS_L1OC_FC, y=YMIN_LABEL+LABEL_YSPACE, text_font_size='9px', text_align='center'),
        Span(location=GLONASS_L3OC_FC,dimension='height', line_color='purple',line_dash='dashed', line_width=0.5),
        Label(text='GLO L1OC', x=GLONASS_L3OC_FC, y=YMIN_LABEL, text_font_size='9px', text_align='center'),
      ]
      #GLONASS FDMA L1OF & L2OF Carriers
      for carrier in range(-7,7):
        #L1OF
        gloL1OFf0 = GLONASS_L1OF_FC + (carrier * GLONASS_L1OF_SPACING)
        freqAnnotationsGlonass.append(Span(location=gloL1OFf0, dimension='height', line_color='purple',line_dash='dashed', line_width=0.2))

        #L2OF
        gloL2OFf0 = GLONASS_L2OF_FC + (carrier * GLONASS_L2OF_SPACING)
        freqAnnotationsGlonass.append(Span(location=gloL2OFf0, dimension='height', line_color='purple',line_dash='dashed', line_width=0.2))

      #GLONASS FDMA labels
      freqAnnotationsGlonass.append(Label(text='GLO L1OF', x=GLONASS_L1OF_FC, y=YMIN_LABEL, text_font_size='9px', text_align='center'))
      freqAnnotationsGlonass.append(Label(text='GLO L2OF', x=GLONASS_L2OF_FC, y=YMIN_LABEL, text_font_size='9px', text_align='center'))

      self.spectrumFigures[block].renderers.extend(freqAnnotationsGPS+freqAnnotationsGalileo+freqAnnotationsGlonass+freqAnnotationsQZSS+freqAnnotationsBeidou)

      #Metadata label
      self.blockMetadataLabels[block] = Div(text=f'NO_DATA', width=PLOT_WIDTH, height=20)

      #Create a column with rows for plot and metadata
      self.blockColumnLayouts[block] = column(row(children=[self.spectrumFigures[block]],sizing_mode="stretch_both"), self.blockMetadataLabels[block])

    #Row layout of columns with plot and additional metadata
    self.doc.add_root(row(children=self.blockColumnLayouts, sizing_mode="stretch_both"))

    print (f"Reading from {inputBuffer}")
    self.ubxScopeQueue = UBXScopeQueue(ser=inputBuffer, eofTimeout=0, onUBXCallback=self.onUBXMessage)
    self.ubxScopeQueue.start()


  #Reset cumulative average when set visible
  def cmaVisibleChangeHandler(self,attr,old,new):
    if new == True:
      for block in range(self.numRfBlocks):
        self.spectrumDataSource.data[f'spectrumCMA_{block}'] = np.zeros(256)

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








