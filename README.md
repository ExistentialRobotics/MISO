
<div align="center">
  <a href="http://erl.ucsd.edu/">
    <img align="left" src="docs/media/erl.png" width="80" alt="erl">
  </a>
  <a href="https://contextualrobotics.ucsd.edu/">
    <img align="center" src="docs/media/cri.png" width="150" alt="cri">
  </a>
  <a href="https://ucsd.edu/">
    <img align="right" src="docs/media/ucsd.png" width="260" alt="ucsd">
  </a>
</div>


# MISO: Multiresolution Submap Optimization for Efficient Globally Consistent Neural Implicit Reconstruction

<p align="center">
  📄 <a href="https://arxiv.org/abs/2504.19104"><strong>Paper</strong></a> |
  🌐 <a href="https://existentialrobotics.org/miso_rss25/"><strong>Project Website</strong></a>
</p>

This is the code repository for MISO, a hierarchical optimization approach that leverages multiresolution submaps to achieve efficient and scalable neural implicit reconstruction. For local SLAM within each submap, we develop a hierarchical optimization scheme with learned initialization that substantially reduces the time needed to optimize the implicit submap features. To correct estimation drift globally, we develop a hierarchical method to align and fuse the multiresolution submaps, leading to substantial acceleration by avoiding the need to decode the full scene geometry. 

<p align="center">
    <a href="https://youtu.be/EADUWRSkOKs?si=d8Rs7uKJreQm6Str">
    <img src="docs/media/miso.png" alt="ROAM">
    </a>
</p>


## Citation

If you found this work useful, we would appreciate if you could cite our work:

- Y. Tian, H. Cao, S. Kim,  N. Atanasov. [**MISO: Multiresolution Submap Optimization for Efficient Globally Consistent Neural Implicit Reconstruction**](https://arxiv.org/pdf/2504.19104). [arXiv:2504.19104](https://arxiv.org/pdf/2504.19104).
 
 ```bibtex
@inproceedings{tian2025miso,
  title={{MISO}: Multiresolution Submap Optimization for Efficient Globally Consistent Neural Implicit Reconstruction},
  author={Tian, Yulun and Cao, Hanwen and Kim, Sunghwan and Atanasov, Nikolay},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025}
}
```

## Acknowledgments

We gratefully acknowledge support from ARL DCIST CRA W911NF-17-2-0181, ONR N00014-23-1-2353, and NSF CCF-2112665 (TILOS).

## License

[BSD License](LICENSE.BSD)
