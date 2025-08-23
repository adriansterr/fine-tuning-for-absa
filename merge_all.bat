setlocal enabledelayedexpansion

REM Create lock file to indicate merge is running
echo MERGING > "D:\Uni\Masterarbeit Code\jakob_finetuning\merge_in_progress.lock"

REM Set paths
set RUN_DIR=D:\Uni\Masterarbeit Code\test\mergekit\run
set MERGE_DIR=D:\Uni\Masterarbeit Code\test\mergekit\merges\trained\llama\meta_llama_full_precision_sauerkraut\evaluate
set SOURCE_TOKENIZER=D:\Uni\Masterarbeit Code\jakob_finetuning\finetuned_models\meta_llama_full_colab_remerge_2

REM Merge all YAMLs
for %%F in ("%RUN_DIR%\*.yaml") do (
    set "YAML=%%~nxF"
    set "NAME=%%~nF"
    set "OUTDIR=%MERGE_DIR%\!NAME!"
    if not exist "!OUTDIR!" mkdir "!OUTDIR!"

    echo Merging %%~nxF...
    mergekit-yaml "%%F" "!OUTDIR!" --allow-crimes --lazy-unpickle --no-copy-tokenizer

    REM Check if tokenizer already exists
    if exist "!OUTDIR!\tokenizer_config.json" (
        echo Tokenizer already exists for %%~nxF, skipping copy...
    ) else (
        echo Copying tokenizer for %%~nxF...
        REM Copy tokenizer files from your working model
        copy "%SOURCE_TOKENIZER%\tokenizer.json" "!OUTDIR!\" 2>nul
        copy "%SOURCE_TOKENIZER%\tokenizer_config.json" "!OUTDIR!\" 2>nul
        copy "%SOURCE_TOKENIZER%\special_tokens_map.json" "!OUTDIR!\" 2>nul
        echo Tokenizer copied for %%~nxF
    )

    echo Completed %%~nxF
)

REM Remove lock file when done
del "D:\Uni\Masterarbeit Code\jakob_finetuning\merge_in_progress.lock"

REM Create completion marker
echo MERGE_COMPLETE > "D:\Uni\Masterarbeit Code\jakob_finetuning\merge_complete.flag"

echo All merges completed with tokenizers!

endlocal