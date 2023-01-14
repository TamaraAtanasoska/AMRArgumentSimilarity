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

To get the results using the S2match metric which is denoted as "structure" in the paper you can pass a `weighting_scheme` argument set to 'structure': 
```
cd repro_repos/amr-metric-suite/py3-Smatch-and-S2match
python smatch/s2match.py -f ../examples/a.txt ../examples/b.txt -cutoff 0.95 -weighting_scheme structure --ms
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

### Parsing the Datasets
To generate amr parses of the datasets one can use `scripts/generate_amr_files.py`. It assumes the dataset is a csv and has two sentence columns to be processed into amr parses. To generate amr files run:
```
python scripts/generate_amr_files.py --data_path_csv <path-to-dataset-csv> --column_name1 <sentence_1_column_name>  --column_name2 <sentence_2_column_name> --out_path <path-to-put-amr-parses>
```
`<path-to-dataset-csv>` should contain the csv with the dataset with the columns indicated as `<sentence_1_column_name>` and  `<sentence_2_column_name>`. The `--column_name` arguments are potional: default names for the columns are `sentence_1` and `sentence_2`. The resulting amr files `amr.src` and `amr.tgt` are put into the folder indicated with `<path-to-put-amr-parses>`. 

Additionally, a batch size can be provided to process the sentences in batches. The default is 5 sentences per batch:

```
python scripts/generate_amr_files.py --data_path_csv <path-to-dataset-csv> --column_name1 <sentence_1_column_name>  --column_name2 <sentence_2_column_name> --batch_size <batch_size> --out_path <path-to-put-amr-parses>
```

## Datasets

### Persuasive Essay Corpus

The original paper [Opitz et al., 2021] uses [Argument Annotated Essays Corpus [Stab & Gurevych, 2017]](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422) for fine-tuning the conclusion T5 generation model. The paper states that “from all premise-conclusion-pairs annotated in this dataset, we retrieved all claims with their annotated premises. In addition, we employ all annotated major claims with their supportive claims as premise-conclusion-pairs” and that “whenever we encounter multiple premises or supportive claims of a single claim, we concatenate them in document order.”

To generate the corpus for fine-tuning T5 summarization on the persuasive essay corpus run:

```
python scripts/generate_persuasive_essay_corpus.py --ann_dir <path-to-dataset>/ArgumentAnnotatedEssays-2.0/brat-project-final --out_path <path-to-output>/premises_conclusions.csv
```

Th script assumes `<path-to-dataset>/ArgumentAnnotatedEssays-2.0` folder contains the contents of the official dataset distribution. 

The script reads the argumentative essay annotation files and extracts major claims, claims, and premises. The script writes to the `<path-to-output>/premises_conclusions.csv` file with 3 columns - `Essay`, `Premises`, `Claim`.

The premises-claim pairs are created as follows: 

1. All claims supporting a major claim are concatenated (separated with `' ### '`) and paired with the major claim. If there are several major claims,  all of the major claims get the same supporting claim sets, e.g `essay0, claim1 ### claim2 ### claim3, major_claim1`
2. All premises supproting a claim are concatenated in the same way and paired with the claim, e.g `essay0, claim1 ### claim2 ### claim3, claim4`
3. All premises supproting a premise are concatenated in the same way and paird with the premies, e.g `essay0, premise1 ### premise2 ### premise3, premise4`

### UPK Dataset
To test the metric on the [UPK Aspect dataset](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/1998) a renaming scheme needs to be applied to obtain binary scores in accordance with the original article.

```
python scripts/rescale_upk_dataset.py --upk_path <path-to-upk-corpus-tsv> --out_file <out-path>/UPK_corpus.csv
```

The `<path-to-upk-corpus-tsv>` should be the UPK argument similarity tsv file distributed from the corpus official website. 

The resulting CSV file is written to `<out-path>/UPK_corpus.csv` and contains the following columns: `topic`, `sentence_1`, `sentence_2`, `regression_label_binary`. The sentence and topic columns are copied form the original dataset; the binary label is 1 if the original label is above 'HS' or 'SS' (*highly similar* or *somewhat similar*) and 0 otherwie. No scale of `regression_label` is available for this dataset, only binary scores.

### BWS Dataset
To test the metric on the [BWS dataset](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2496) a rescaling scheme needs to be applied to obtain binary scores.

```
python scripts/rescale_bws_dataset.py --bws_path <path-to-bws-corpus-csv> --out_file <out-path>/BWS_corpus.csv
```

The `<path-to-bws-corpus-csv>` should be the BWS argument similarity csv file distributed from the corpus official website. 

The resulting CSV file is written to `<out-path>/BWS_corpus.csv` and contains the following columns: `topic`, `sentence_1`, `sentence_2`, `regression_label_binary`, `regression_label`. The sentence, topic, and regression_label (*score*) columns are copied form the original dataset; the binary label is 1 if the original label is above 0.5 and 0 otherwie.

### AFS Dataset
To test the metric on the [AFS dataset](https://nlds.soe.ucsc.edu/node/44), a rescaling scheme needs to be applied, as the metric is developped for [0,1] similarity schores, and AFS features [0,5] similarity scores.

To merge the three topics of the argument facet similarity dataset into one csv, rescaling the scores from [0,5] to [0,1], along with binary {0, 1} labels run the following comand:

```
python scripts/rescale_AFS_dataset.py --afs_path <path-to-afs-corpus> --out_file <out-path>/AFS_corpus.csv
```

The `<path-to-afs-corpus>` folder should contain the 3 argument similarity csv files distributed from the corpus official website. 

The resulting CSV file is written to `<out-path>/AFS_corpus.csv` and contains the following columns: `topic`, `sentence_1`, `sentence_2`, `regression_label_binary`, `regression_label`. The sentence columns are copied form the original dataset; the topic is ‘GM’, ‘GC’ or ‘DP’ depemding on argument topic; the binary label is 1 if the original label is 4 or 5 and 0 otherwie; and the scaled label is min-max scaled to 0-1 values, scaling being applied per topic.

## Fine-tuning and summarisation

The original paper uses conclusions generated by a T5 model, fine-tuned on the Persuasive Essay Corpus discussed above to enhance the s2match scores. The authors also mention trying summarisation, but opting out for the fine-tuned model because of better results. As no models or code were available for neither fine-tuning or summarisation, we made our own attempt of reproducing them, documented below. 

### Weights & Biases

We have introduced [Weights & Biases](https://wandb.ai/site) as platform support to visualize and keep track of our experiments. You could take advantage of this integration by adding the option ```--wandb``` to the fine-tuning or generation commands. 

If you decide to use the option, Weights & Biases will ask you to log in so you can have access to the visualizations and the logging of the runs. You will be prompted to pick an option about how to use W&B, and logging in will subsequently require your W&B API key. It might be more practical for you to already finish this setup before starting the training runs with this option. You can read [here](https://docs.wandb.ai/ref/cli/wandb-login) how to do that from the command line. Creating an account before this step is necessary. 

It is necessay to initialise the entity and project name: [example](https://github.com/TamaraAtanasoska/AMR_ArgumentSimilarity/blob/main/conclusion_generation/fine_tuning/fine-tune.py#L96). You can edit this line to add your own names, and learn more about these settings in the [W&B documentation](https://docs.wandb.ai/ref/python/init). 

### Fine-tuning a T5 model to perfom conclusion generation 

The code for the fine tuning can found in [conclusion_generation/fine_tuning](conclusion_generation/fine_tuning). It is based almost fully on [this](https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb#scrollTo=OKRpFvYhBauC) notebook, with some changes to the way the [W&B](https://wandb.ai/site) parameters are handled and cleaning up of the deprecation errors. The original notebook was listed on the [T5 Huggingface website](https://huggingface.co/docs/transformers/model_doc/t5). 

To fine-tune, you just need to pass the path to the location file:
```
cd conclusion_generation/fine_tuning
python fine-tune.py --data_path <path-to-dataset>
```
The fine-tuned model for inference will be saved at ```conclusion_generation/fine_tuning/models/conclusion_generation_model.pth```.

## Extra

We looked for the best hyperparameters for the fine-tuning with a W&B sweep. Besides running the command below, you will need to add entity and project name as with the W&B experiment tracking in the code. In order to do that, search for a ```sweep_id``` occurence in [fine_tune.py](conclusion_generation/fine_tuning/fine-tune.py). We have very limited computational resources, so the sweep is with very small ranges and all strictly defined. 

```
cd conclusion_generation/fine_tuning
python fine-tune.py --data_path <path-to-dataset> --wandb_sweep
```
