# HSA-NRL
This repo is the official implementation of our paper "Hard Sample Aware Noise Robust Learning for Histopathology Image Classification".

## Using instructions
`step1.py` is the label correction phase, and `NSHE.py` is the NSHE phase. 

Run `step1.py` first to generate the "Almost clean dataset". Then run `NSHE.py` to train the model by the "Almost clean dataset".

Take Chaoyang dataset as an example:

First, run `python step1.py --dataset chaoyang` and get the "Almost clean dataset" file named "chaoyang_15_step1_filtered_dataset_iter1_over.p".

Then, run `python NSHE.py --dataset chaoyang --forget_rate 0.01 --pickle_path chaoyang_15_step1_filtered_dataset_iter1_over.p` to train the model by the generated "Almost clean dataset".

## Data
[Chaoyang(Google drive)](https://drive.google.com/open?id=1xsrHjn-WyHGazYtpMqHo9h2w349eYCYO&authuser=bupt.ai.cz%40gmail.com&usp=drive_fs)

## Citation

```
@article{zhuhard,
  title={Hard Sample Aware Noise Robust Learning for Histopathology Image Classification},
  author={Zhu, Chuang and Chen, Wenkai and Peng, Ting and Wang, Ying and Jin, Mulan},
  journal={IEEE transactions on medical imaging}
}
```

## Contact

Wenkai Chen
- email: wkchen@bupt.edu.cn
- wechat: cwkyiyi

Chuang Zhu
- email: czhu@bupt.edu.cn

If you have any questions, please contact us directly.

## Additional Info
Some parts of our code are borrowed from [the official Co-teaching implementation](https://github.com/IsaacChanghau/CoTeaching).

We are continuing to add the using instructions of our code.

## Acknowledgements
- Thanks Chaoyang hospital for dataset annotation
- Thanks Yi Chen for the design of Fig. 3 in our paper.
