# Evaluations of DNABERT2 on GUE dataset with disabled heads

Download (GUE dataset)[https://drive.google.com/file/d/1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2/view] and put it in this folder after cloning.
Make sure inside this GUE folder is the task folders, i.e. EMP, mouse, prom, etc. You may have to move it out after extracting the zip.

Run `./run_dnabert2.sh .` to finetune and evaluate DNABERT2 on the GUE dataset.

In train.py, line 304 is where the logic for running the model validation for each head disabled at a time, using `disable_head_hook`.

There may be an error involving the transformers library being unable to pickle a `mappingproxy` object, and this error originates from line 4030 of `path_to_python/Lib/site-packages/transformers/trainer.py`.

I personally just added a try/except in the file like this, but it may be possible that this does not occur at all on your end:

4029    # Good practice: save your training arguments together with the trained model
4030    try:
4031        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
4032    except TypeError as e:
4033        logger.warning(f"Exception encountered while trying to save training args: {str(e.with_traceback(e.__traceback__))}")


The outputs for the evals should be in `disabled_head_evals/output/dnabert2_{run_name}/results/`.

# Please let me know if anything doesn't work I haven't run this in a while because it takes so long LOL