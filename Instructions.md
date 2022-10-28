# To work with annotation files in .TextGird format:
This document will introduce how to use the cough annotation files.<br />

## Dependencies
-Programming language: Our current version only supports **Python** 3. <br />
-Python packages: Textgrid module for importing .TextGrid files, which can be found at [https://github.com/kylebgorman/textgrid].
-Visaulization tool: .TextGrid files can be also imported to PRAAT ([https://www.fon.hum.uva.nl/praat/]) for visualization with the raw audio file.

## How to visualize
Import the raw annotation file together with the .wav/.flac audio file into PRAAT. Select both of them and click 'view and edit', you should be able to see the cough waveform and the corresponding annotations of different cough phases.

## How to import to Python for further analysis
```
import textgrid
tg = textgrid.TextGrid.fromFile(ANNOTIONFILE_PATH)
```
This should return an ```object``` which has a hierarchical architecture to store all the annotation information. An annotation object should have three **tiers**: tier1 includes the onset and offset of each phase in the opened cough recording; tier2 and tier3 contains the number of inhalation and expulsion in the opened cough recordings. 
```
phase_tier = tg.tiers[0]
num_cough = tg.tiers[1].intervals[0].mark
number_inhale = tg.tiers[2].intervals[0].mark
```
Tier1 has multiple **intervals**, where each **interval** corresponds to a single cough phase. The onset and offset of each phase can be accessed as follows: <br />
Say we want to access the first phase in the recoridng:
```
phase_num = 0
phase_name = tg.tiers[0].intervals[phase_num].mark
phase_onset = tg.tiers[0].intervals[phase_num].minTime
phase_offset = tg.tiers[0].intervals[phase_num].maxTime
```
You now know the onset and offset (in second) of the phase. Iterating through all **intervals** in **tier1**, you should be able to get all annotations of the opened cough recording.
