# Segmented phases from COVID-19 cough recordings

## Overview and background
Cough is an important symptom of numerous respiratory diseases, including COVID-19. While different cough phases (i.e., inhalation, compression, and expulsion) have been shown to be related to different pathological origins, existing cough-based COVID-19 detection systems rely on the entire cough recording, thus such phase-related characteristics are overlooked. In this study, our aim is two-fold. First, we have annotated over 1,250 cough recordings from two publicly-available cough sound databases, thus providing the research community with fine-grained cough phase labels. Next, we extract a number of temporal and acoustic features from each cough phase and test their usefulness and complementarity for COVID-19 detection. Experiments show the importance of cough phase segmentation, not only for improved COVID-19 detection, but also for the development of models that are interpretable and can better generalize across datasets.

## What does this repo contain?
This repo contains: <br />
1. Cough phase annotation files of ComParE and DiCOVA2 cough recordings.
2. Scripts to reproduce results shown in our paper **"On the importance of different cough phases for COVID-19 detection"**, which can be found here: https://www.techrxiv.org/articles/preprint/On_the_importance_of_different_cough_phases_for_COVID-19_detection/21382176/1

## How to access, visualize, and/or use the annotation file to segment cough recordings?
You can find a detailed instruction in ```Instructions.md```. We have included an example showing you how to import files, visualize them with PRAAT, and segment recordings into phases.

## Where do I get access to the original recordings and labels?
For data privacy purposes, we cannot share the cough sound data in this repository. But the sound files as well as the corresponding COVID-19 labels can be obtained upon approval from the organizers of ComParE and DiCOVA2 challenges. The contact info as well as details about these challenges can be found in the following papers: <br />
A summary of the ComParE challenge: https://arxiv.org/abs/2202.08981 <br />
The parental dataset of ComParE (Cambridge sound database): https://openreview.net/forum?id=9KArJb4r5ZQ <br />
DiCOVA2 challenge: https://arxiv.org/abs/2110.01177 

## How to generate results shown in the paper?
If you are only interested in reproducing the results shown in this paper, we encourage you to simply run the  ```covid_prediction.py``` from inside of the ```scripts``` folder. Try different features stored in the ```feature``` folder as well as their combinations, you should be able to get the same results.

## How to generate features and visualize the group difference?
For generating the acoustic features, check ```acoustic_feature_extraction.py```. <br />
For generating the temporal features (and visualize histograms, etc.), check ```temporal_feature_extraction.py```.

## Reference
For more details, you can refer to our paper https://www.techrxiv.org/articles/preprint/On_the_importance_of_different_cough_phases_for_COVID-19_detection/21382176/1
. If you are using the annotation files or the scripts in this repo, please make a reference to this paper.

## Acknowledgement and disclaimer
We would like to thank the organizers of DiCOVA2 and ComParE challenges for collecting and sharing the sound data. The aforementioned organizations do not bear any responsibility for the analysis and results presented in our paper. All results and interpretation only represent the view of the authors. We also thank our annotators Zack and Hong for spending their time on the annotation!

## Author and contact info
This repo is created by Yi Zhu in Oct,2022. If you have any questions, feel free to contact at **Yi.Zhu@inrs.ca**.
