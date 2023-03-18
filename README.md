# CloverLeaf HIP

A version of CloverLeaf ported to HIP using the `hipify-perl` tool.
In this case, many manual changes were needed after running the tool because the original port involves three languages which is probably too much for some perl regex script.

## Compilation

This is an example to make Cloverleaf on a Cray machine:

```
HIP_HOME=/opt/rocm-4.5.1/hip/ make COMPILER=CRAY AMDGCN_ARCH=gfx908 -j 6 C_MPI_COMPILER=cc MPI_COMPILER=ftn 
```

* COMPILER is the same as with the other implementations.
* AMDGCN_ARCH is the architecture - in the form of gfxXXX
* Job number only affects parallel build of *.cu files
* (C_)MPI_COMPILER affects which compiler to use
