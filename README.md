# Reproduction study: Argument Similarity with AMR

This project reproduces the paper "Explainable Unsupervised Argument Similarity Rating with Abstract Meaning Representation and Conclusion Generation"(Opitz et al., 2021) [(link)](project_docs/OpitzEtAl21.pdf). 

To reproduce the results of the study, two repositories originally written by the author are required: [the more general repository](https://github.com/flipz357/amr-metric-suite) that contains AMR metrics and the [paper-specific repository](https://github.com/Heidelberg-NLP/amr-argument-sim) that contains the AMR metric for argument similarity. As we focus on the results obtained in the paper, we provide both repositories cloned in [```repro_repos/```](repro_repos/). Additionally, we use the [AMR parser](https://github.com/bjascob/amrlib) as used in the paper as a library to turn sentences into graphs.

It will also include extensions to the original article. We plan to test the same method on two other argument similarity corpora (BWS Argument Similarity Corpus ([Thakur et al., 2021](https://arxiv.org/abs/2010.08240)) and Argument Facet Similarity Dataset ([Misra et al., 2016](https://aclanthology.org/W16-3636/))). We also plan to explore how conclusion generation fine-tuning affects the results and how the length of the premises interacts with conclusion generation contributions.

This project is a so-called "Project Module", part of the [Cognitive Systems](https://www.uni-potsdam.de/en/studium/what-to-study/master/masters-courses-from-a-to-z/cognitive-systems) Master program at the [University of Potsdam](https://www.uni-potsdam.de/en/university-of-potsdam). Contributors: [Tamara Atanasoska](https://github.com/TamaraAtanasoska), [Emanuele De Rossi](https://github.com/EmanueleDeRossi1) and [Galina Ryazanskaya](https://github.com/flying-bear).

## Setup

### Environment recreation 

In the folder ```setup/``` you can find the respective environment replication and package requirements files. There are two options:

  1. You can run ```pip install -r setup/requirements.txt``` to install the necessary packages in your existing environment.

  2. If you are using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to manage your virtual environments, you can replicate and activate the full exact environment with the following commands:

   ```
   conda env create --name <name> --file setup/conda.yml
   conda activate <name>
   ```
   
You might encounter an issue with the package ```mkl_fft``` if you are using conda. You could run the following to install it:
```
conda install -c intel mkl_fft
```

## Generating the S2match scores and evaluating with the AMR metric

### Downloading word vectors

S2match needs word vectors to perform the calculation. There is a script included to download Glove word vectors. 

```
cd repro_repos/amr-metric-suite/
./download_glove.sh
```

### Generating S2match scores

To get the results using the S2match metric which is denoted as "standard" in the paper you can run: 
```
cd repro_repos/amr-metric-suite/py3-Smatch-and-S2match
python smatch/s2match.py -f ../examples/a.txt ../examples/b.txt -cutoff 0.95 --ms
```

This command uses the example files that are already in the folder, they can be replaced with any other file in the right format. The cutoff parameter's value of 0.95 corresponds to the value used in the paper. A high cutoff parameter allows the score to increase only for (near-) paraphrase concepts.

You would need to save the output in a file to be able to use it to evaluate with the AMR metric below. You could either copy-paste the output or run the following complete command: 
```
cd repro_repos/amr-metric-suite/py3-Smatch-and-S2match
python smatch/s2match.py -f ../examples/a.txt ../examples/b.txt -cutoff 0.95 --ms > s2match_scores_standard.txt
```

To get the results using the S2match metric which is denoted as "concept" in the paper you can pass a `weighting_scheme` argument set to 'concept': 
```
cd repro_repos/amr-metric-suite/py3-Smatch-and-S2match
python smatch/s2match.py -f ../examples/a.txt ../examples/b.txt -cutoff 0.95 -weighting_scheme concept --ms
```

### Evaluating with the AMR metric

In the [```sim_preds/```](repro_repos/amr-argument-sim/scripts/sim_preds) folder, there are various predictions stored. To test-evaluate all of them with the AMR metric you can run the command below. You will see the output as the command runs.  
```
cd repro_repos/amr-argument-sim/scripts/
./evaluate_all.sh
```
What the script does, is use all the files that are in the similarity predictions folder to evaluate. If you would like to use a file of your own, you can use the [```evaluate.py```](repro_repos/amr-argument-sim/scripts/evaluate.py) script. 

## Using the AMR parser

If you have set up your environment with [our section](#environment-recreation) about it above, you will already have all the packages installed. If you haven't and you would like to install only the relevant ones for this parser, please take a look at the parser's [installation guide](https://amrlib.readthedocs.io/en/latest/install/). If you only need the parser, it is a better idea to clone the code from the [original repository](https://github.com/bjascob/amrlib) too. 

After installing all the required packages by any of the means, run: 
```
pip install amrlib
```
This will install the library. However, in order to parse, you will also need to pick and download a model to do that with. All the models currently available are found in [this repository](https://github.com/bjascob/amrlib-models). 

We picked to use the [oldest T5-based parse model](https://github.com/bjascob/amrlib-models/releases/tag/model_parse_t5-v0_1_0) available, to be able to get as comparable results with the ones in the paper as possible. 

Once you have picked the model, you need to download it, extract it and rename it. More information can be found [here](https://amrlib.readthedocs.io/en/latest/install/#install-the-models). On Windows, it would be easier to just download the zip file, unzip it and rename the folder instead of the last linking command. 
```
pip show amrlib #copy path where the package is stored 
cd <path-to-where-the-package-is-stored> #copy path from the output of the command above

mkdir data
cd data
tar xzf <model-filename> #copy file here before running command
ln -snf <model-filename>  model_stog
```

To test if the parser is working and the installation is correct, you can run: 
```
cd scripts
python ./test_parse.py 
```

## Datasets

### Persuasive Essay Corpus

The original paper [Opitz et al., 2021] uses [Argument Annotated Essays Corpus [Stab & Gurevych, 2017]](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422) for fine-tuning the conclusion T5 generation model. The paper states that “from all premise-conclusion-pairs annotated in this dataset, we retrieved all claims with their annotated premises. In addition, we employ all annotated major claims with their supportive claims as premise-conclusion-pairs” and that “whenever we encounter multiple premises or supportive claims of a single claim, we concatenate them in document order.”

To generate the corpus for fine-tuning T5 summarization on the persuasive essay corpus run:

```
python scripts/generate_persuasive_essay_corpus.py --ann_dir <PATH_TO_DATASET>/ArgumentAnnotatedEssays-2.0/brat-project-final --out_path <PATH_TO_OUTPUT>/premises_conclusions.csv
```

Th script assumes `<PATH_TO_DATASET>/ArgumentAnnotatedEssays-2.0` folder contains the contents of the official dataset distribution. 

The script reads the argumentative essay annotation files and extracts major claims, claims, and premises. The script writes to the `<PATH_TO_OUTPUT>/premises_conclusions.csv` file with 3 columns - `Essay`, `Premises`, `Claim`.

The premises-claim pairs are created as follows: 

1. All claims supporting a major claim are concatenated (separated with `' ### '`) and paired with the major claim. If there are several major claims,  all of the major claims get the same supporting claim sets, e.g `essay0, claim1 ### claim2 ### claim3, major_claim1`
2. All premises supproting a claim are concatenated in the same way and paired with the claim, e.g `essay0, claim1 ### claim2 ### claim3, claim4`
3. All premises supproting a premise are concatenated in the same way and paird with the premies, e.g `essay0, premise1 ### premise2 ### premise3, premise4`

### AFS Dataset
To test the metric on the [AFS dataset](https://nlds.soe.ucsc.edu/node/44), a rescaling scheme needs to be applied, as the metric is developped for [0,1] similarity schores, and AFS features [0,5] similarity scores.

To merge the three topics of the argument facet similarity dataset into one csv, rescaling the scores from [0,5] to [0,1], along with binary {0, 1} labels run the following comand:

```
python scripts/rescale_AFS_dataset.py --afs_path <PATH_TO_AFS_CORPUS> --out_file <OUT_PATH>/AFS_corpus_combined.csv
```

The `<PATH_TO_AFS_CORPUS>` folder should contain the 3 argument similarity csv files distributed from the corpus official website. 

The resulting CSV file is written to `<OUT_PATH>/AFS_corpus_combined.csv` and contains the following columns: `regression_label`, `sentence_1`, `sentence_2`, `topic`, `regression_label_binary`, `regression_label_scaled`. The first three columns are copied form the original dataset; the topic is ‘GM’, ‘GC’ or ‘DP’ depemding on argument topic; the binary label is 1 if the original label is above 4 or 5 and 0 otherwie; and the scaled label is min-max scaled to 0-1 values, scaling being applied per topic.
