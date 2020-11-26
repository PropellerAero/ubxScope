#!/usr/bin/env python3

import ubx
from ubx import parseUBXMessage, UBXManager
from pathlib import Path
from queue import Queue
from functools import partial

import sys
import numpy as np
import argparse
import csv

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Legend, Span, Label, Div, TapTool

TOOLTIPS = [
    ("PSD", "$y dB @ $x Hz"),
]

TOOLS = "pan,wheel_zoom,box_zoom,reset,save,tap"

# Averaging Window Length
TIME_AVERAGING_WINDOW_LENGTH = 10  # data frames (i.e defined by epoch rate)

# Bin Width
# This is defined in SPAN but for convenience of plotting setup
# it is hard coded here
SPAN_BIN_COUNT = 256

# Axes bounds
YMAX = 180
YMIN = 30
YMIN_LABEL = YMIN + 5
YMAX_LABEL = YMAX - 5
LABEL_YSPACE = 2

# Arbitrary initial plot frame size, for referencing annotation
# positions before the plot is scaled
PLOT_WIDTH = 400
PLOT_HEIGHT = 400

# Common 'band' frequencies
X1_FC = 1575420000
X2_FC = 1227600000
X5A_FC = 1176450000
X5B_FC = 1207140000
X6_FC = 1278750000

# GNSS Signals (Hz)
# Multiple signals sharing a frequency for each system are not shown
# The signal identifier for the 'open' signal is added
# Where multiple open signals are present, no identifier is added

# GPS
GPS_L1_CA_P_C_M_FC = X1_FC  # L1 C/A, P, C, M
GPS_L2_P_CM_CL_M_FC = X2_FC  # L2 P, CL, CM, M
GPS_L5_I_Q_FC = X5A_FC  # L5 I,Q

# Glonass FDMA
GLONASS_L1_OF_SF_FC = 1602000000  # L1 OF,SF
GLONASS_L1_OF_SF_SPACING = 562500
GLONASS_L2_OF_SF_FC = 1246000000  # L2 OF,SF
GLONASS_L2_OF_SF_SPACING = 437500

# Glonass CDMA
GLONASS_L1_OC_SC_FC = 1600995000  # L1 OC,SC
GLONASS_L2_OC_SC_FC = 1248060000  # L2 OC,SC
GLONASS_L3_OC_SC_FC = 1202025000  # L3 OC,SC

# Galileo
GALILEO_E1_I_Q_FC = X1_FC  # E1 I,Q
GALILEO_E5A_I_Q_FC = X5A_FC  # E5a I,Q
GALILEO_E5B_I_Q_FC = X5B_FC  # E5b I,Q
GALILEO_E5_ALTBOC_FC = 1191795000  # E5 Altboc (a + b)
GALILEO_E6_I_Q_PRS_FC = X6_FC  # E6 I,Q,PRS

# Beidou
BEIDOU_B1_I_Q_FC = 1561098000  # B1 I,Q
BEIDOU_B1_C_A_FC = X1_FC  # B1 C,a
BEIDOU_B2_I_Q_B_FC = X5B_FC  # B2 I,Q,b
BEIDOU_B2_A_FC = X5A_FC  # B2 a
BEIDOU_B3_I_Q_A_FC = 1268520000  # B3 I,Q,A

# SBAS
SBAS_L1_FC = X1_FC
SBAS_L5_FC = X5A_FC

# QZSS
QZSS_L1_CA_C_SAIF_FC = X1_FC  # L1 CA, C, SAIF
QZSS_L2_CM_CL_FC = X2_FC  # L2 CM, CL
QZSS_L5_I_Q_FC = X5A_FC  # L5 I,Q
QZSS_E6_LEX_FC = X6_FC  # E6 LEX

# Be quiet on errors.


class UBXScopeQueue(UBXManager):
    def __init__(self, ser, debug=False, eofTimeout=None, onUBXCallback=None):
        self._queue = Queue()
        # Reflects the has-a queue's get() and empty() methods
        self.empty = self._queue.empty
        self.onUBXCallback = onUBXCallback

        super(UBXScopeQueue, self).__init__(
            ser=ser, debug=debug, eofTimeout=eofTimeout)

    def onUBXError(self, msgClass, msgId, errMsg):
        return

    def onNMEA(self, msg):
        return

    def onNMEAError(self, msg):
        return

    def onUBX(self, msg):
        if msg.__class__.__name__ in ['SPAN', 'PVT']:
            self.onUBXCallback(msg, msg.__class__.__name__)
        # else:
            # print(f'Unhandled: {msg.__class__.__name__}')


class UBXScope:
    def __init__(self, inputBuffer):
        self.numRfBlocks = 2

        # Metadata stores
        # Per block
        self.spectrumMetadata = [{'pga': 0, }
                                 for block in range(self.numRfBlocks)]
        # Global
        self.ubxMetadata = {'timeUTC': ""}

        # CSV Output
        self.csvFile = open('out.csv', 'w')
        self.writer = csv.writer(self.csvFile)

        # Setup Plot
        self.doc = curdoc()
        self.doc.title = "UBX Scope"

        # Per-block arrays
        self.spectrumFigures = [self.numRfBlocks, None]
        self.blockMetadataLabels = [self.numRfBlocks, None]
        self.blockColumnLayouts = [self.numRfBlocks, None]
        self.spectrumAvgBuffers = [self.numRfBlocks, None]
        self.spectrumDataSources = [self.numRfBlocks, None]
        self.selectionLabels = [self.numRfBlocks, None]
        self.selectionMarkers = [self.numRfBlocks, None]
        self.selectionLabelData = [self.numRfBlocks, None]
        # Add a figure for each block
        for block in range(self.numRfBlocks):

            # One Data Source for each figure, shared per-plot
            self.spectrumDataSources[block] = ColumnDataSource(data={
                'spectrumBinCenterFreqs': np.zeros(SPAN_BIN_COUNT),
                'spectrumMax': np.zeros(SPAN_BIN_COUNT),
                'spectrumAvg': np.zeros(SPAN_BIN_COUNT),
                'spectrum': np.zeros(SPAN_BIN_COUNT)
            })

            # Moving Average Buffer
            self.spectrumAvgBuffers[block] = {
                'buffer': np.zeros((TIME_AVERAGING_WINDOW_LENGTH, SPAN_BIN_COUNT)),
                'index': 0,  # Use the ndarray as a circular buffer
                'filled': 1  # Buffer fullness
            }

            self.selectionLabelData[block] = {
                'dataSourceIndex': 0,
                'frequency': 0,
                'maxPower': 0,
                'avgPower': 0,
                'power': 0
            }

            figure_ = figure(title=f"UBX SPAN Block {block+1}",
                             output_backend="webgl",
                             y_range=(YMIN, YMAX),
                             tooltips=TOOLTIPS,
                             tools=TOOLS,
                             plot_width=PLOT_WIDTH,
                             plot_height=PLOT_HEIGHT)

            # Add instantaneous, avg, and max line plots
            spectrum = figure_.line(source=self.spectrumDataSources[block],
                                    x='spectrumBinCenterFreqs',
                                    y='spectrum',
                                    line_width=1,
                                    line_color='blue'
                                    )

            spectrumMax = figure_.line(source=self.spectrumDataSources[block],
                                       x='spectrumBinCenterFreqs',
                                       y='spectrumMax',
                                       line_width=1,
                                       line_color='red'
                                       )
            spectrumAvg = figure_.line(source=self.spectrumDataSources[block],
                                       x='spectrumBinCenterFreqs',
                                       y='spectrumAvg',
                                       line_width=1,
                                       line_color='green'
                                       )

            #spectrum.data_source.on_change('selected', self.selectCallback)
            #spectrum.on_change('line_indices', self.selectCallback)
            self.spectrumDataSources[block].selected.on_change(
                'line_indices', partial(self.lineSelectCallback, block=block))

            # Handlers to reset aggregates when visibility is changed
            spectrumAvg.on_change('visible', partial(
                self.avgVisibleChangeHandler, block=block))
            spectrumMax.on_change('visible', partial(
                self.maxVisibleChangeHandler, block=block))

            # Label Axes
            figure_.xaxis.axis_label = "Frequency (Hz)"
            figure_.yaxis.axis_label = "Received Power dB (Unref)"

            # Legend
            legend = Legend(items=[
                ("PSD", [spectrum]),
                ("Max PSD", [spectrumMax]),
                ("Avg PSD", [spectrumAvg]),
            ], location="center",
                click_policy="hide")
            figure_.add_layout(legend, 'left')

            self.spectrumFigures[block] = figure_

            # Centre Frequencies
            # GPS
            freqAnnotationsGPS = [
                Span(location=GPS_L1_CA_P_C_M_FC, dimension='height',
                     line_color='orange', line_dash='dashed', line_width=0.5),
                Label(text='GPS L1', x=GPS_L1_CA_P_C_M_FC, y=YMIN_LABEL,
                      text_font_size='9px', text_align='center', text_font_style='bold'),
                Span(location=GPS_L2_P_CM_CL_M_FC, dimension='height',
                     line_color='orange', line_dash='dashed', line_width=0.5),
                Label(text='GPS L2', x=GPS_L2_P_CM_CL_M_FC, y=YMIN_LABEL,
                      text_font_size='9px', text_align='center', text_font_style='bold'),
                Span(location=GPS_L5_I_Q_FC, dimension='height',
                     line_color='orange', line_dash='dashed', line_width=0.5),
                Label(text='GPS L5', x=GPS_L5_I_Q_FC, y=YMIN_LABEL,
                      text_font_size='9px', text_align='center')
            ]

            freqAnnotationsGalileo = [
                Span(location=GALILEO_E1_I_Q_FC, dimension='height',
                     line_color='green', line_dash='dashed', line_width=0.5),
                Label(text='GAL E1', x=GALILEO_E1_I_Q_FC, y=YMIN_LABEL+LABEL_YSPACE,
                      text_font_size='9px', text_align='center', text_font_style='bold'),
                Span(location=GALILEO_E5A_I_Q_FC, dimension='height',
                     line_color='green', line_dash='dashed', line_width=0.5),
                Label(text='GAL E5b', x=GALILEO_E5A_I_Q_FC, y=YMIN_LABEL +
                      LABEL_YSPACE, text_font_size='9px', text_align='center'),
                Span(location=GALILEO_E5B_I_Q_FC, dimension='height',
                     line_color='green', line_dash='dashed', line_width=0.5),
                Label(text='GAL E5a', x=GALILEO_E5B_I_Q_FC, y=YMIN_LABEL+LABEL_YSPACE,
                      text_font_size='9px', text_align='center', text_font_style='bold'),
                Span(location=GALILEO_E5_ALTBOC_FC, dimension='height',
                     line_color='green', line_dash='dashed', line_width=0.5),
                Label(text='GAL E5 ALTBOC', x=GALILEO_E5_ALTBOC_FC, y=YMIN_LABEL +
                      LABEL_YSPACE, text_font_size='9px', text_align='center'),
                Span(location=GALILEO_E6_I_Q_PRS_FC, dimension='height',
                     line_color='green', line_dash='dashed', line_width=0.5),
                Label(text='GAL E6', x=GALILEO_E6_I_Q_PRS_FC, y=YMIN_LABEL +
                      LABEL_YSPACE, text_font_size='9px', text_align='center')
            ]

            # Beidou
            freqAnnotationsBeidou = [
                Span(location=BEIDOU_B1_I_Q_FC, dimension='height',
                     line_color='red', line_dash='dashed', line_width=0.5),
                Label(text='BDS B1 I/Q', x=BEIDOU_B1_I_Q_FC, y=YMIN_LABEL+(3*LABEL_YSPACE),
                      text_font_size='9px', text_align='center', text_font_style='bold'),
                Span(location=BEIDOU_B1_C_A_FC, dimension='height',
                     line_color='red', line_dash='dashed', line_width=0.5),
                Label(text='BDS B1 C/a', x=BEIDOU_B1_C_A_FC, y=YMIN_LABEL +
                      (3*LABEL_YSPACE), text_font_size='9px', text_align='center'),
                Span(location=BEIDOU_B2_A_FC, dimension='height',
                     line_color='red', line_dash='dashed', line_width=0.5),
                Label(text='BDS B2 a', x=BEIDOU_B2_A_FC, y=YMIN_LABEL +
                      (3*LABEL_YSPACE), text_font_size='9px', text_align='center'),
                Span(location=BEIDOU_B2_I_Q_B_FC, dimension='height',
                     line_color='red', line_dash='dashed', line_width=0.5),
                Label(text='BDS B2 I/Q/b', x=BEIDOU_B2_I_Q_B_FC, y=YMIN_LABEL+(3*LABEL_YSPACE),
                      text_font_size='9px', text_align='center', text_font_style='bold'),
                Span(location=BEIDOU_B3_I_Q_A_FC, dimension='height',
                     line_color='red', line_dash='dashed', line_width=0.5),
                Label(text='BDS B3 I/Q/A', x=BEIDOU_B3_I_Q_A_FC, y=YMIN_LABEL +
                      (3*LABEL_YSPACE), text_font_size='9px', text_align='center'),
            ]

            freqAnnotationsQZSS = [
                Span(location=QZSS_L1_CA_C_SAIF_FC, dimension='height',
                     line_color='yellow', line_dash='dashed', line_width=0.5),
                Label(text='QZSS L1', x=QZSS_L1_CA_C_SAIF_FC, y=YMIN_LABEL+(4*LABEL_YSPACE),
                      text_font_size='9px', text_align='center', text_font_style='bold'),
                Span(location=QZSS_L2_CM_CL_FC, dimension='height',
                     line_color='yellow', line_dash='dashed', line_width=0.5),
                Label(text='QZSS L2', x=QZSS_L2_CM_CL_FC, y=YMIN_LABEL+(4*LABEL_YSPACE),
                      text_font_size='9px', text_align='center', text_font_style='bold'),
                Span(location=QZSS_L5_I_Q_FC, dimension='height',
                     line_color='yellow', line_dash='dashed', line_width=0.5),
                Label(text='QZSS L5', x=QZSS_L5_I_Q_FC, y=YMIN_LABEL +
                      (4*LABEL_YSPACE), text_font_size='9px', text_align='center'),
                Span(location=QZSS_E6_LEX_FC, dimension='height',
                     line_color='yellow', line_dash='dashed', line_width=0.5),
                Label(text='QZSS E6 LEX', x=QZSS_E6_LEX_FC, y=YMIN_LABEL +
                      (4*LABEL_YSPACE), text_font_size='9px', text_align='center')
            ]

            freqAnnotationsGlonass = [
                Span(location=GLONASS_L1_OC_SC_FC, dimension='height',
                     line_color='purple', line_dash='dashed', line_width=0.5),
                Label(text='GLO L1 CDMA', x=GLONASS_L1_OC_SC_FC, y=YMIN_LABEL +
                      LABEL_YSPACE, text_font_size='9px', text_align='center'),
                Span(location=GLONASS_L2_OC_SC_FC, dimension='height',
                     line_color='purple', line_dash='dashed', line_width=0.5),
                Label(text='GLO L2 CDMA', x=GLONASS_L2_OC_SC_FC, y=YMIN_LABEL +
                      LABEL_YSPACE, text_font_size='9px', text_align='center'),
                Span(location=GLONASS_L3_OC_SC_FC, dimension='height',
                     line_color='purple', line_dash='dashed', line_width=0.5),
                Label(text='GLO L3 CDMA', x=GLONASS_L3_OC_SC_FC,
                      y=YMIN_LABEL, text_font_size='9px', text_align='center'),
            ]
            # GLONASS FDMA L1OF & L2OF Carriers
            for carrier in range(-7, 9):
                # L1OF
                gloL1OFf0 = GLONASS_L1_OF_SF_FC + \
                    (carrier * GLONASS_L1_OF_SF_SPACING)
                freqAnnotationsGlonass.append(Span(
                    location=gloL1OFf0, dimension='height', line_color='purple', line_dash='dashed', line_width=0.2))

                # L2OF
                gloL2OFf0 = GLONASS_L2_OF_SF_FC + \
                    (carrier * GLONASS_L2_OF_SF_SPACING)
                freqAnnotationsGlonass.append(Span(
                    location=gloL2OFf0, dimension='height', line_color='purple', line_dash='dashed', line_width=0.2))

            # GLONASS FDMA labels
            freqAnnotationsGlonass.append(Label(text='GLO L1 FDMA', x=GLONASS_L1_OF_SF_FC,
                                                y=YMIN_LABEL, text_font_size='9px', text_align='center', text_font_style='bold'))
            freqAnnotationsGlonass.append(Label(text='GLO L2 FDMA', x=GLONASS_L2_OF_SF_FC,
                                                y=YMIN_LABEL, text_font_size='9px', text_align='center', text_font_style='bold'))

            self.spectrumFigures[block].renderers.extend(
                freqAnnotationsGPS+freqAnnotationsGalileo+freqAnnotationsGlonass+freqAnnotationsQZSS+freqAnnotationsBeidou)

            # Selection Markers (Initially Hidden)
            self.selectionMarkers[block] = Span(
                location=0, dimension='height', line_color='blue', line_dash='dotted', line_width=2, visible=False)
            self.spectrumFigures[block].add_layout(
                self.selectionMarkers[block])

            # Metadata label
            self.blockMetadataLabels[block] = Div(text=f'No Data', width=PLOT_WIDTH, height=20)
            self.selectionLabels[block] = Div(text='No Selection: ', height=20)

            # Create a column with rows for plot and metadata
            self.blockColumnLayouts[block] = column(
                row(children=[self.spectrumFigures[block]],
                    sizing_mode="stretch_both"),
                self.blockMetadataLabels[block],
                self.selectionLabels[block]
            )

        # Items to draw not specific to one block
        self.timeLabel = Div(text='No Time', height=20)

        # Row layout of columns with plot and additional Metadata
        self.doc.add_root(column(children=[
                                 row(children=[self.timeLabel],
                                     sizing_mode="stretch_width"),
                                 row(children=self.blockColumnLayouts,
                                     sizing_mode="stretch_both")
                                 ], sizing_mode="stretch_both"))

        print(f"Reading from {inputBuffer}")
        self.ubxScopeQueue = UBXScopeQueue(
            ser=inputBuffer, eofTimeout=0, onUBXCallback=self.onUBXMessage)
        self.ubxScopeQueue.start()

    def updateSelectionLabel(self, block, visible=False):
        if visible:
            data = self.selectionLabelData[block]
            # Assign to a text div
            self.selectionLabels[block].text = f"Frequency: {data['frequency']} | PSD: {data['power']}dB | Max: {data['maxPower']}dB | Avg: {data['avgPower']}dB"
        else:
            self.selectionLabels[block].text = f'No Selection'

    def lineSelectCallback(self, attr, old, new, block):
        # This is received as a string representation of an array but numpy can parse it
        # If more than one point selected find the mean and scale to the nearest int
        indices = np.array(new)
        if len(indices) > 0:
            index = int(np.floor(np.mean(np.array(new))))

            # Set current data values for selection
            self.selectionLabelData[block]['dataSourceIndex'] = index
            self.selectionLabelData[block]['frequency'] = self.spectrumDataSources[block].data['spectrumBinCenterFreqs'][index]
            self.selectionLabelData[block]['maxPower'] = self.spectrumDataSources[block].data['spectrumMax'][index]
            self.selectionLabelData[block]['avgPower'] = self.spectrumDataSources[block].data['spectrumAvg'][index]
            self.selectionLabelData[block]['power'] = self.spectrumDataSources[block].data['spectrum'][index]

            # And update the text label
            self.updateSelectionLabel(block, visible=True)

            # And set marker frequency
            self.selectionMarkers[block].location = self.selectionLabelData[block]['frequency']
            self.selectionMarkers[block].visible = True

        # No Selection
        else:
            self.updateSelectionLabel(block, visible=False)
            self.selectionMarkers[block].visible = False

    # Reset cumulative average when set visible
    def avgVisibleChangeHandler(self, attr, old, new, block):
        # when set visible
        if new == True:
            # reset buffer indexes
            self.spectrumAvgBuffers[block]['index'] = 0
            self.spectrumAvgBuffers[block]['filled'] = 1
            # Set first row of the buffer and the current displayed average to the current spectrum
            self.spectrumAvgBuffers[block]['buffer'][self.spectrumAvgBuffers[block]
                                                     ['index'], :] = self.spectrumDataSources[block].data['spectrum']
            self.spectrumDataSources[block].data['spectrumAvg'] = self.spectrumDataSources[block].data['spectrum']

    # Reset spectrum max when set visible
    def maxVisibleChangeHandler(self, attr, old, new, block):
        if new == True:
            self.spectrumDataSources[block].data['spectrumMax'] = np.zeros(
                SPAN_BIN_COUNT)

    def updateSpectrumPlots(self, spectrumDataBlocks):
        # Update spectrum data
        for block, spectrumData in enumerate(spectrumDataBlocks):
            self.spectrumDataSources[block].data = spectrumData

        # Spectrum metadata
        for index, blockMetadata in enumerate(self.spectrumMetadata):
            pgaGain = blockMetadata['pgaGain']
            self.blockMetadataLabels[index].text = f'PGA Gain: {pgaGain}dB'

        # Update the selection marker text, if the marker is set
        for block in range(self.numRfBlocks):
            if self.selectionMarkers[block].visible:

                # Set current data values for selection
                # Use the stored index value for the selected frequency
                index = self.selectionLabelData[block]['dataSourceIndex']
                self.selectionLabelData[block]['frequency'] = self.spectrumDataSources[block].data['spectrumBinCenterFreqs'][index]
                self.selectionLabelData[block]['maxPower'] = self.spectrumDataSources[block].data['spectrumMax'][index]
                self.selectionLabelData[block]['avgPower'] = self.spectrumDataSources[block].data['spectrumAvg'][index]
                self.selectionLabelData[block]['power'] = self.spectrumDataSources[block].data['spectrum'][index]

                # And update the text label
                self.updateSelectionLabel(block, visible=True)

        # Global Metadata
        time = self.ubxMetadata['timeUTC']
        self.timeLabel.text = f'Time: {time}'

    def onUBXMessage(self, msg, msgClass):

        if msgClass == 'SPAN':

            # Process new spectrum data
            newSpectrumDataBlocks = [self.numRfBlocks, None]
            for block in range(msg.numRfBlocks):
                newSpectrumData = {}

                # Centre Frequencies
                newSpectrumData['spectrumBinCenterFreqs'] = msg.spectra[block]['spectrumBinCenterFreqs']

                # PSD bin data
                newSpectrumData['spectrum'] = msg.spectra[block]['spectrum']

                # Calculate PSD max
                newSpectrumData['spectrumMax'] = np.maximum(
                    newSpectrumData['spectrum'], self.spectrumDataSources[block].data['spectrumMax'])

                # Calculate Moving Average
                # Replace row at index, to avoid push/pop. Order/wrapping doesn't matter unless weighting is applied
                self.spectrumAvgBuffers[block]['buffer'][self.spectrumAvgBuffers[block]
                                                         ['index'], :] = newSpectrumData['spectrum']

                # calculate column-wise (time) mean of for all filled (non-zero) buffer rows
                newSpectrumData['spectrumAvg'] = np.sum(
                    self.spectrumAvgBuffers[block]['buffer'][0:self.spectrumAvgBuffers[block]['filled'], :], axis=0) / self.spectrumAvgBuffers[block]['filled']

                #Write CSV
                meta = ["spectrumAvg", str(block)]
                line = meta + [str(round(w, 2)) for w in newSpectrumData['spectrumAvg']]
                self.writer.writerow(line)

                # Additional metadata for annotations
                self.spectrumMetadata[block]['pgaGain'] = msg.spectra[block]['pga']

                # Add spectrum for each block to one array to return, to assign to the data sources
                newSpectrumDataBlocks[block] = newSpectrumData

                # Increment indexes for moving average windows
                self.spectrumAvgBuffers[block]['index'] = self.spectrumAvgBuffers[block]['index'] + 1
                # Circular buffer wrap
                if self.spectrumAvgBuffers[block]['index'] >= TIME_AVERAGING_WINDOW_LENGTH:
                    self.spectrumAvgBuffers[block]['index'] = 0

                # Increment buffer fullness till full
                if (self.spectrumAvgBuffers[block]['filled'] < TIME_AVERAGING_WINDOW_LENGTH):
                    self.spectrumAvgBuffers[block]['filled'] = self.spectrumAvgBuffers[block]['filled'] + 1

            self.doc.add_next_tick_callback(
                partial(self.updateSpectrumPlots, spectrumDataBlocks=newSpectrumDataBlocks))

        if msgClass == 'PVT':
            self.ubxMetadata['timeUTC'] = msg.UTC


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=argparse.FileType('rb'), required=False)
args = parser.parse_args()
if (args.file):
    sys.stderr.write(f"-- Reading from {args.file}\n\n")
    scope = UBXScope(args.file)
else:
    sys.stderr.write(f"-- Reading from stdin\n\n")
    scope = UBXScope(sys.stdin.buffer)
