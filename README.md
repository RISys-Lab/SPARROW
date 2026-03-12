<div align="center">
<h1> SPARROW<img src='assets/sparrow.png' align="center" width="5%">: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs (CVPR 2026) </h1>

> [**SPARROW**](https://arxiv.org/abs/2404.13013)
> 
> **Mohamad Alansari, Naufal Suryanto, Divya Velayudhan, Sajid Javed, Naoufel Werghi, and Muzammal Naseer**
> 
><a href="https://arxiv.org/abs/2404.13013"><img src='https://img.shields.io/badge/arXiv-SPARROW-red' alt='Paper PDF'></a>
><a href='https://sparrow-mllm.github.io/'><img src='https://img.shields.io/badge/Project_Page-SPARROW-green' alt='Project Page'></a>
><a href='https://huggingface.co/RISys-Lab/sparrow-finetune'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
><a href='https://huggingface.co/datasets/RISys-Lab/sparrow-dataset'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow'></a>

<img src='assets/Fig1.png' align="center" width="80%">
<p align="left">Comparison of temporal consistency and initialization quality in video object segmentation. 
    (a) The baseline method suffers from temporal drift, leading to inconsistent segmentation of the same object across frames. 
    (b) Noisy or unstable initialization propagates segmentation errors through subsequent frames. 
    (c) Our proposed Target-Specific Tracked Feature mitigates drift by maintaining consistent object grounding over time. 
    (d) The Dual-Prompt Initialization strategy improves segmentation precision and stability during early frames.
</p>
</div>


## Contents
- [Install](#installation)
- [Model](#model-weights)
- [Data](#prepare-data)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)


## Performance
State-of-the-art and consistent improvements across referring video object segmentation, video visual grounding, and grounded conversation generation benchmarks.

<table> <thead> <tr> <th rowspan="2">Method</th> <th colspan="2">MeViS</th> <th colspan="2">RVOS</th> <th rowspan="2">VidSTG<br>mIoU</th> <th rowspan="2">VideoGCG<br>mIoU</th> </tr> <tr> <th>val<br>J&amp;F</th> <th>val<sup>u</sup><br>J&amp;F</th> <th>Ref-YTVOS<br>J&amp;F</th> <th>Ref-DAVIS17<br>J&amp;F</th> </tr> </thead> <tbody> <tr align="center"> <td>UniPixel</td> <td>53.1</td> <td>59.7</td> <td>70.5</td> <td>74.2</td> <td>41.25</td> <td>52.0</td> </tr> <tr align="center" style="background-color: #EAF3FF;"> <td>UniPixel + SPARROW</td> <td><b>54.4</b></td> <td><b>60.7</b></td> <td><b>70.7</b></td> <td><b>76.4</b></td> <td><b>46.74</b></td> <td><b>54.5</b></td> </tr> <tr align="center"> <td>GLUS</td> <td>51.3</td> <td>59.8</td> <td>67.3</td> <td>72.9</td> <td>29.92</td> <td>45.86</td> </tr> <tr align="center" style="background-color: #EAF3FF;"> <td>GLUS + SPARROW</td> <td><b>53.2</b></td> <td><b>61.9</b></td> <td><b>69.1</b></td> <td><b>75.5</b></td> <td><b>35.17</b></td> <td><b>47.91</b></td> </tr> <tr align="center"> <td>VideoGLaMM</td> <td>45.2</td> <td>48.5</td> <td>66.8</td> <td>69.5</td> <td>39.66</td> <td>62.34</td> </tr> <tr style="background-color: #ADD8E6;" align="center"> <th>VideoGLaMM + SPARROW</th> <th>47.5</th> <th>57.4</th> <th>68.9</th> <th>76.8</th> <th>45.06</th> <th>65.59</th> </tr> </tbody> </table>
