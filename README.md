# SDeeEff - A PyTorch SDF learner

A set of Python PyTorch programs to create a Signed Distance Field (SDF), for use with a raymarching GLSL shader.

A neural network - Siren - is trained on a particular .obj file. This object is repeatedly sampled by the program, using the mesh-to-sdf library. Points in the space are sampled with this library, returning the distance to the object's surface. The network learns the distance from the sampled point, until all we need is the network weights to estimate our SDF.

The weights are then exported as text, which can be pasted into the sdf.glsl file included. Or, if you prefer, one can use ShaderToy or PoshBrolly to view the SDF.

Based on the following:
* [https://procegen.konstantinmagnus.de/neural-network-sdfs](https://procegen.konstantinmagnus.de/neural-network-sdfs)
* [https://www.vincentsitzmann.com/siren/](https://www.vincentsitzmann.com/siren/)
* [https://ieeexplore.ieee.org/document/8954065/](https://ieeexplore.ieee.org/document/8954065/)
* [https://www.shadertoy.com/view/wtVyWK](https://www.shadertoy.com/view/wtVyWK)
* [https://www.youtube.com/watch?v=8pwXpfi-0bU](https://www.youtube.com/watch?v=8pwXpfi-0bU)
* [https://drive.google.com/drive/folders/13-ks7iyLyI0vcS38xq1eeFdaMdfNlUC8](https://drive.google.com/drive/folders/13-ks7iyLyI0vcS38xq1eeFdaMdfNlUC8)
*[https://pypi.org/project/mesh-to-sdf/](https://pypi.org/project/mesh-to-sdf/)

This repository is mostly just a rehash of the excellent work done by [https://www.youtube.com/@suricrasia](Blackle Mori).

## Requirements and setup

I use the virtualenv model to setup all the requirements.

    python -m venv venv
    source ./venv/bin/activate
    pip install -r requirements.txt

The main libraries are PyTorch and mesh-to-sdf.

## Running

To train, go with:

    python train.py --obj gormley4.obj 

Then cut and paste the result into the sdf.glsl

I use glslCanvas in vscode to look at the SDF. Seems like it works alright. ctrl-shift-p to run it. 

## Currently

Gormley5.obj is no longer unit sized. I commented out the bit to get it all into a unit sphere as I think that might be causing detail issues. Trying to expand out a bit.