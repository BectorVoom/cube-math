use cubecl::prelude::*;

macro_rules! t {
    ($m:ident, $f:ident) => {
        pub mod $m {
            use super::*;
            const ZERO: $f = 0.0;
            const FALSE: bool = false;
            #[cube]
            pub fn g(x: $f) -> $f {
                let mut out = ZERO;
                let mut flag = FALSE;
                if x > 1.0 {
                    out = x;
                    flag = true;
                }
                select(flag, out, -out)
            }
        }
    };
}
t!(d, f64);

#[cube(launch_unchecked)]
fn k(inp: &Array<f64>, out: &mut Array<f64>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = d::g(inp[ABSOLUTE_POS]);
    }
}

fn main() {
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    let inp = client.create(cubecl::bytes::Bytes::from_elems(vec![2.0f64, 0.5]));
    let o = client.empty(16);
    unsafe {
        k::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts(inp, 2),
            ArrayArg::from_raw_parts(o.clone(), 2),
        );
    }
    println!("{:?}", f64::from_bytes(&client.read_one(o).unwrap()));
}
