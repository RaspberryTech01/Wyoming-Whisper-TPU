# Wyoming WhisperTPU Python

This python package is designed to run on the Sophon BM1684X, the SoC TPU present on the Fogwise Airbox made by Radxa. This project has been forked from Radxa's own Whisper implementation and developed to integrate the Wyoming protocol by Rhasspy - the parts of the Faster Whisper project by Rhasspy has been integrated into this one as well.

The most notable change compared to the original forked project is the model is loaded in and stays in memory until the Wyoming protcol sends in a new audio file for the system to process. This speeds up the responses considerably.

## Environment
The codebase is expected to be compatible with Python 3.8-3.11 and recent PyTorch versions. The codebase also depends on a few Python packages, most notably OpenAI's tiktoken for their fast tokenizer implementation.

Configure the virtual environment.
**It is essential to create a virtual environment to avoid potential conflicts with other applications.** For instructions on using a virtual environment, refer to [this guide](Virtualenv_usage.md).

```bash
cd whisper-TPU_py
python3 -m virtualenv .venv 
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt 
python3 setup.py install

It also requires the command-line tool `ffmpeg` to be installed on your system, which is available from most package managers:
```bash
# if you use a conda environment
conda install ffmpeg
 
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg 
```

## Download BModels
You have the option of either creating your own BModels, or downloading them using Radxa's instructions found here: https://github.com/radxa-edge/TPU-Edge-AI/blob/main/radxa_docs/EN/Whisper.md.

On a Fogwise Airbox, your ``/`` directory will not have enough space for these models. I placed mine in ``/data/whisper_models``, but it will require sudo permissions.

```bash
sudo mkdir /data/whisper_models
cd /data/whisper_models
sudo wget https://github.com/radxa-edge/TPU-Edge-AI/releases/download/Whisper/tar_downloader.sh
sudo bash tar_downloader.sh
sudo tar -xvf Whisper_bmodel.tar.gz
```

## Command-line usage
### TPU mode
Please disable debug info first:
```bash
export LOG_LEVEL=-1
```

Please set your sophon library path:
```bash
export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH
```

Default model is `small`, start using whisper-TPU with `bmwhisper` in the directory `whisper-TPU_py`

Or you can set the absolute path of bmodel dir like this `--bmodel_dir [bmodel_dir]`, and `bmwhisper` can be used anywhere:
```bash
bmwhisper --bmodel_dir /your/path/to/bmodel_dir
```
You can change the model by adding `--model [model_name]`:
```bash
bmwhisper --model medium
```
Model available now:
* base
* small
* medium
You can change the chip mode by adding `--chip_mode soc`, default is `pcie`:
```bash
bmwhisper demo.wav --chip_mode soc
```

**Example usage:** `bmwhisper  --model base  --language en --bmodel_dir /data/whisper_models/Whisper_bmodel/bmodel/ --chip_mode soc`