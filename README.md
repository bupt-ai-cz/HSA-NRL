# HSA-NRL [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hard-sample-aware-noise-robust-learning-for/image-classification-on-chaoyang)](https://paperswithcode.com/sota/image-classification-on-chaoyang?p=hard-sample-aware-noise-robust-learning-for) ![visitors](https://visitor-badge.glitch.me/badge?page_id=bupt-ai-cz.HSA-NRL) [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Codes%20and%20Data%20for%20Our%20Paper:%20"Hard:%20Sample%20Aware%20Noise%20Robust-Learning%20for%20Histopathology%20Image%20Classification"%20&url=https://github.com/bupt-ai-cz/HSA-NRL)  
This repo is the official implementation of our paper ["Hard Sample Aware Noise Robust Learning for Histopathology Image Classification"](https://ieeexplore.ieee.org/document/9600806).

<div align="center">
<img src="img/TMI-2021.png" height="400" width="800">
</div>

## News
- ⚡(2021-11-20): Chaoyang dataset was released [HERE](https://bupt-ai-cz.github.io/HSA-NRL/). 

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
[Chaoyang](https://bupt-ai-cz.github.io/HSA-NRL/)

- Chaoyang dataset contains 1111 normal, 842 serrated, 1404 adenocarcinoma, 664 adenoma, and 705 normal, 321 serrated, 840 adenocarcinoma, 273 adenoma samples for training and testing, respectively. (Notes: "0" means normal, "1" means serrated, "2" means adenocarinoma, and "3" means adenoma in our dataset files.)


## Using instructions
- **Notes:** `step1.py` is the label correction phase, `NSHE.py` is the NSHE phase. 

- **Getting started:**

    Run `step1.py` first to generate the "Almost clean dataset". Then run `NSHE.py` to train the model by the "Almost clean dataset".

    Take Chaoyang dataset as an example:

    First, run `python step1.py --dataset chaoyang` and get the "Almost clean dataset" file named "chaoyang_15_step1.p".

    Then, run the command below to train the model by the generated "Almost clean dataset".

    `python NSHE.py --dataset chaoyang --forget_rate 0.01 --pickle_path chaoyang_15_step1.p` 

    (Remember to modify the dataset path before using.)

## License

This project is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree to our license terms bellow:

1. That you include a reference to our paper in any work that makes use of the data/code. For research papers, cite our preferred publication; for other media cite our preferred publication or link to [our github project](https://github.com/bupt-ai-cz/HSA-NRL).
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


## Acknowledgements

- Thanks Chaoyang hospital for dataset annotation.
