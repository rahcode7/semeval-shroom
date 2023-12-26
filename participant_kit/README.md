# SemEval 2024 Shared Task 6: SHROOM - Participant kit

This participant kit contains a few useful scripts intended to simplify the submission process during the evaluation phase.
It contains:
 - a baseline system showcasing data loading and expected output format;
 - a script for checking the well-formedness of system outputs;
 - the scoring program current used on the CodaLab competition website.

NB: the scoring program as used on the evaluation platform is susceptible to change up till the start of the evaluation phase, in the event of necessary bug fixes or updates.

## Baseline system
The baseline system is based on a simple prompt retrieval approach, derived from SelfCheck-GPT. It uses an open-source Mistral instruction-finetuned model as its core component.
The baseline system was tested on a RTX 3080 Nvidia GPU, with 16GB of VRAM. It may require adjustments to work on other hardwares.

The key points this baseline system is intended to illustrate include:
 - how to load data, using either built-in python libraries (`json`) or third party libraries (HuggingFace's `datasets`);
 - how to produce an output with the format required for the scoring program;
 - file naming conventions (participants should make sure that the output filename for a track matches with the filename of the corresponding reference file).

## Output checker script
The output checker script is intended as a general first pass to ensure that the model's output generally correspond to the requirements of the scoring program. 
Note that it may not flag all potential issues.

To evaluate the well-formedness of a validation-split output, use the `--is_val` flag.
In line with codalab expectations, participants should provide as inputs paths to directories containing their submission file(s), rather than path to files.

## Scoring program
The scorer script requires the reference data to compute scores. The test reference data will only be made available after the end of the evaluation phase, on February 1st.
Note that the scorer script may reject a submission that was marked as correct by the output format checker script.

To evaluate the performances on validation data, use the `--is_val` flag. The reference data file names must be modified to not include version numbering when present.
In line with codalab expectations, participants should provide as inputs paths to directories containing their submission file(s) and the corresponding reference files.

## Baseline Results
Trial data: 
rho: 0.5971938519429985
acc: 0.7375

===============================    
Validation data:
acc_aware:0.7065868263473054
rho_aware:0.46095769289667354

acc_agnostic:0.6492985971943888
rho_agnostic:0.3801408906585249 



