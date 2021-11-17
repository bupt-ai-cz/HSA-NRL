# HSA-NRL
This repo is the official implementation of our paper ["Hard Sample Aware Noise Robust Learning for Histopathology Image Classification"](https://ieeexplore.ieee.org/document/9600806).

## Citation
If you use this code/data for your research, please cite our paper ["Hard Sample Aware Noise Robust Learning for Histopathology Image Classification"](https://ieeexplore.ieee.org/document/9600806).

```
@article{zhuhard,
  title={Hard Sample Aware Noise Robust Learning for Histopathology Image Classification},
  author={Zhu, Chuang and Chen, Wenkai and Peng, Ting and Wang, Ying and Jin, Mulan},
  journal={IEEE transactions on medical imaging}
}
```


## Data
[Chaoyang(Google drive)](https://drive.google.com/open?id=1xsrHjn-WyHGazYtpMqHo9h2w349eYCYO&authuser=bupt.ai.cz%40gmail.com&usp=drive_fs)

DigestPath2019(processed) and Camelyon16(processed) are coming soon.

## Using instructions
`step1.py` is the label correction phase, and `NSHE.py` is the NSHE phase. 

Run `step1.py` first to generate the "Almost clean dataset". Then run `NSHE.py` to train the model by the "Almost clean dataset".

Take Chaoyang dataset as an example:

First, run `python step1.py --dataset chaoyang` and get the "Almost clean dataset" file named "chaoyang_15_step1_filtered_dataset_iter1_over.p".

Then, run the command below to train the model by the generated "Almost clean dataset".

`python NSHE.py --dataset chaoyang --forget_rate 0.01 --pickle_path chaoyang_15_step1_filtered_dataset_iter1_over.p` 

(Remember to modify the dataset path before using.)

## License

This project is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree to our license terms bellow:

1. That you include a reference to the Chaoyang Dataset in any work that makes use of the dataset. For research papers, cite our preferred publication; for other media cite our preferred publication or link to [our github project](https://github.com/bupt-ai-cz/HSA-NRL).
2. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data).
3. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
4. That all rights not expressly granted to you are reserved by us.




## Contact

Wenkai Chen
- email: wkchen@bupt.edu.cn
- wechat: cwkyiyi

Chuang Zhu
- email: czhu@bupt.edu.cn
- homepage: https://teacher.bupt.edu.cn/zhuchuang/zh_CN/index.htm

If you have any questions, please contact us directly.

## Additional Info

Some parts of our code are borrowed from the [official Co-teaching implementation](https://github.com/bhanML/Co-teaching).

We are continuing to add the using instructions of our code.

## Acknowledgements

- Thanks Chaoyang hospital for dataset annotation.
- Thanks Yi Chen for the design of Fig. 3 in our paper.


