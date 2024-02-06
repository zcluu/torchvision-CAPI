# torchvision-CAPI

Torchvision based on C++

## How to Use?

### Build

You can build torchvision-CAPI with `python setup.py build`.

After building, you'll find the `Shared Object File` in `build/lib.linux-x86_64-*/extension.cpython-310-x86_64-linux-gnu.so`. You can use `ln -s` to create a symbolic link to `extension.so`.

### Run

#### Python

```Python
import torch
import extension

help(extension)
```

#### Rust

**01. Dependencies**

```toml
# Please use the same version as torch.
tch = "0.14.0"
torch-sys = "0.14.0"
```

If you want to write a Python extension in Rust, you need to add `pyo3` and `pyo3-tch`.

Tips: `PyTensor` in `pyo3-tch` is not compatible with `torch::Tensor` in `libtorch` and this repository. If you want to use this repository, you need to use `torch_sys::C_tensor`.

**02. Declare the function**

You need to use `extern "C" {}` to declare the function.

```rust
#[link(name = "extension")]
extern "C" {
    fn crop(
        img: *const C_Tensor,
        top: i64,
        left: i64,
        height: i64,
        width: i64
    ) -> C_Tensor;
    ...
}

fn test() {
    let img = torch::rand(3, 32, 32);
    let img = crop(img, 1, 1, 8, 8);
    println!("img: {}", img);
}
```

### Test

`pytest test_py/test_all.py`
