## U2GAN: Unsupervised Universal Self-Attention Network for Graph Classification

This program provides the implementation of our unsupervised graph embedding model U2GAN as described in [the paper]():

        @article{Nguyen2019U2GAN,
          author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dinh Phung},
          title={{Unsupervised Universal Self-Attention Network for Graph Classification}},
          journal={arXiv preprint arXiv:1909.11855},
          year={2019}
          }
  
Please cite the paper whenever U2GAN is used to produce published results or incorporated into other software. As a free open-source implementation, U2GAN is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

U2GAN is free for non-commercial use and distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA) License. 

<p align="center">
	<img src="https://github.com/daiquocnguyen/U2GAN/blob/master/U2GAN.png">
</p>

## Usage

### Requirements
- Python 3
- Tensorflow >= 1.13
- Tensor2tensor >= 1.9
- LIBLINEAR https://www.csie.ntu.edu.tw/~cjlin/liblinear/

### Training

Regarding the unsupervised U2GAN:

	$ git clone https://github.com/cjlin1/liblinear.git
	
	$ cd liblinear
	
	liblinear$ make
	
	liblinear$ git clone https://github.com/daiquocnguyen/U2GAN.git
	
	liblinear$ cd U2GAN
	
	U2GAN$ unzip dataset.zip

	U2GAN$ python train_u2GAN_noPOS.py --dataset COLLAB --batch_size 512 --ff_hidden_size 1024 --num_neighbors 8 --num_sampled 512 --num_epochs 50 --num_hidden_layers 4 --learning_rate 0.00005 --model_name COLLAB_bs512_dro05_1024_8_idx0_4_3
	
	U2GAN$ python train_u2GAN_noPOS.py --dataset DD --batch_size 512 --degree_as_tag --ff_hidden_size 1024 --num_neighbors 4 --num_sampled 512 --num_epochs 50 --num_hidden_layers 3 --learning_rate 0.00005 --model_name DD_bs512_dro05_1024_4_idx0_3_3

Regarding the supervised U2GAN:

	U2GAN$ python train_U2GAN_Sup.py --dataset IMDBBINARY --batch_size 4 --ff_hidden_size 1024 --fold_idx 1 --num_neighbors 8 --num_sampled 512 --num_epochs 50 --num_hidden_layers 4 --learning_rate 0.0005 --model_name IMDBBINARY_bs4_fold1_dro05_1024_8_idx0_4_1
	
	U2GAN$ python train_U2GAN_Sup.py --dataset MUTAG --batch_size 4 --degree_as_tag --ff_hidden_size 1024 --fold_idx 7 --num_neighbors 16 --num_sampled 512 --num_epochs 50 --num_hidden_layers 5 --learning_rate 0.0005 --model_name MUTAG_bs4_fold7_dro05_1024_16_idx0_5_1
	
**Parameters:** 

`--learning_rate`: The initial learning rate for the Adam optimizer.

`--batch_size`: The batch size.

`--dataset`: Name of dataset.

`--num_epochs`: The number of training epochs.

`--num_hidden_layers`: The number T of timesteps.

`--fold_idx`: The index of fold in 10-fold validation.

**Notes:**

See command examples in `command_examples.txt` and `u2GAN_scripts.zip` for the unsupervised U2GAN; and `u2GAN_scripts_NoPOS_Supervised.zip` for the supervised U2GAN.

Only use `train_u2GAN_noPOS_REDDIT.py` and `eval_REDDIT.py` for a large collection of graphs such as REDDIT if having OOM or problems with Tensorflow when running `train_u2GAN_noPOS.py`. See `reddit_commands.txt`. 
