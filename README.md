## Neural Material Baseline

---

Neural implicit filed gets more attention in the computer vision and graphics community since 2019. 
It can represent the complex images, geometry or NeRF (Neural Radiance Field) in a compact way. 
For rendering, some pioneer works have been done in the neural material representation.

However, there are many challenges in the material representation for researchers and engineers:
- The material is high-dimensional, mostly in HDR space rather than LDR space, sensitive to the network init. method, may have color bias, etc.
- The material have many different types (e.g., BRDF, BTDF, SVBRDF, BTF, BSSRDF) with high-frequency specular (e.g., glint), maybe layered, anisotropic, etc.
- It require *eval* and *sample* two operations to support the rendering pipeline
- The grazing angles effects are tough to model
- The neural material (network) requires integrate to the rendering pipeline (renderer)
- Compared to audios, images, NeRF or SDF (signed distance field), generating data for material fitting is non-trivial

In this project, we aim to build a very simple baseline for the neural material representation (overfitting).
It is a good starting point for the material representation, and we hope it can be a good resources for the rookie in this field :)

### Features
- [x] Neural material fitting with a simple MLP
- [x] Neural material rendering with mitsuba3
- [x] Sampling with normalization flow (disk domain)

### Material Type
Currently, we mainly focus on the BRDF supported by [mitsuba3](https://www.mitsuba-renderer.org/)


### Results

The results are shown in the following figures. The first row is the network output, and the second row is the ground truth.
Obviously, there are some differences between the network output and the ground truth, especially in the grazing angles regions.

[img](assets/network_res.png)

[img](assets/gt.png)


### Related work 

Some related works and resources about neural materials can be found in [Awesome-Neural-Materials](https://github.com/bearprin/Awesome-Neural-Materials)


### Citation
If you find our work helpful for your research, please consider giving a star ⭐