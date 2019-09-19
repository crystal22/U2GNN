## U2GAN: Unsupervised Universal Self-Attention Network for Graph Classification

This program provides the implementation of our unsupervised graph embedding model U2GAN as described in [the paper]():

        @InProceedings{Nguyen2019U2GAN,
          author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dinh Phung},
          title={{Unsupervised Universal Self-Attention Network for Graph Classification}},
          booktitle={...},
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
- Tensorflow >= 1.6
- Tensor2tensor >= 1.9
- LIBLINEAR https://www.csie.ntu.edu.tw/~cjlin/liblinear/

### Training
Command examples to run the program:

	$ python train_u2GAN_noPOS.py --dataset COLLAB --batch_size 512 --ff_hidden_size 1024 --num_neighbors 8 --num_sampled 512 --num_epochs 50 --num_hidden_layers 4 --learning_rate 0.00005 --model_name COLLAB_bs512_dro05_1024_8_idx0_4_3
	
	$ python train_u2GAN_noPOS.py --dataset DD --batch_size 512 --degree_as_tag --ff_hidden_size 1024 --num_neighbors 4 --num_sampled 512 --num_epochs 50 --num_hidden_layers 3 --learning_rate 0.00005 --model_name DD_bs512_dro05_1024_4_idx0_3_3
	
**Parameters:** 

`--learning_rate`: The initial learning rate for the Adam optimizer.

`--batch_size`: The batch size.

`--dataset`: Name of dataset.

`--num_epochs`: The number of training epochs.

`--num_hidden_layers`: The number T of timesteps.
