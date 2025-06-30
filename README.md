# SDeeEff - A PyTorch SDF learner

A set of Python PyTorch programs to create a Signed Distance Field (SDF), for use with a raymarching GLSL shader.

A neural network - Siren - is trained on a particular .obj mesh file. This object is repeatedly sampled by the program, using the mesh-to-sdf library. Points in the space are sampled with this library, returning the distance to the object's surface. The network learns the distance from the sampled point, until all we need is the network weights to estimate our SDF.

The weights are then exported as text, which can be pasted into the sdf.glsl file included. Or, if you prefer, one can use ShaderToy or PoshBrolly to view the SDF.

The majority of this work is based on the work of [Blackle Mori](https://github.com/blackle).

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

    python train.py --obj <path to object file> 

The *current_fragment.glsl* will contain the final network and a shader to visualise it. 

There are a number of options you can take a look at, at the bottom of the train.py script. I've found the defaults seem to work best.