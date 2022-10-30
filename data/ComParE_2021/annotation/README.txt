Cough Annotation Guideline

Background: Classify COVID vs non-COVID with cough events

Protocol: Each participant (healthy/COVID) was asked to voluntarily cough for several times. Each .wav corresponds to one participant, which is supposed to contain one or multiple cough events.

Why to annotate: Other than cough events, other sound might also get recorded such as noise. We want to keep only the information that we need, which is the cough related events.

Goal: To annotate each .wav recording to label different sound types, including:
	- cough (the exhalation period)
	- inhale (the inhalation period that usually happens before cough)
	- compress (the preparation stage that is usually between inhalation and cough)
	- throatclear (throat clear sound)
	- silence (silence part in the recording)
	- noise (noise part in the recording)

Steps: Praat is used for annotation
	1. Import .wav file to Praat
	2. Click 'Annotate' to create .TextGrid object
	3. Create Tier1, name it 'type'
		- Mark the onset and offset to separate each sound type
		- Label each type of sound with the given names above
	4. Create Tier2, name it 'numberofcough'
		- Input the number of cough events
	5. Create Tier3, name it 'numberofinhale'
		- Input the number of inhalations
	6. Save .TextGrid Object and name it as follows
		- Example: Train_001.TextGrid
	6. Save all .TextGrid files in one folder for the sake of further analysis

-----------------------------

ComParE cough annotations (biased recordings removed from training and development set):
training set -> 273
devel set -> 222
test set -> 208