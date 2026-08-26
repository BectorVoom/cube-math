use cube_math::cube::fma::{split, two_product, two_sum};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn k(inp: &Array<f64>, out: &mut Array<f64>) {
    if ABSOLUTE_POS < 1usize {
        let x = inp[0];
        let (hi, lo) = split(x);
        out[0] = hi;
        out[1] = lo;
        let (s, e) = two_sum(inp[1], inp[2]);
        out[2] = s;
        out[3] = e;
        let (p, pe) = two_product(inp[3], inp[4]);
        out[4] = p;
        out[5] = pe;
        // The raw pieces, so a failure says which identity was rewritten.
        let c = x * 134217729.0f64;
        out[6] = c;
        out[7] = c - x;
        out[8] = c - (c - x);
        let a = inp[1];
        let b = inp[2];
        let ss = a + b;
        let bb = ss - a;
        out[9] = bb;
        out[10] = a - (ss - bb);
        out[11] = b - bb;
    }
}

fn run<R: Runtime>(name: &str) {
    let client = R::client(&Default::default());
    let x = 1.0 + f64::EPSILON;
    let a = 1.0f64;
    let b = f64::EPSILON / 2.0;
    let p1 = 1.0 + f64::EPSILON;
    let p2 = 1.0 - f64::EPSILON / 2.0;
    let inp = client.create(cubecl::bytes::Bytes::from_elems(vec![x, a, b, p1, p2]));
    let outh = client.empty(16 * 8);
    unsafe {
        k::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(inp, 5),
            ArrayArg::from_raw_parts(outh.clone(), 16),
        );
    }
    let o = client.read_one(outh).unwrap();
    let g = f64::from_bytes(&o);
    let c = x * 134217729.0f64;
    let hi = c - (c - x);
    let lo = x - hi;
    let ss = a + b;
    let bb = ss - a;
    let e = (a - (ss - bb)) + (b - bb);
    let want = [
        hi, lo, ss, e,
        p1 * p2, 0.0,
        c, c - x, c - (c - x),
        bb, a - (ss - bb), b - bb,
    ];
    let names = ["split.hi", "split.lo", "sum.s", "sum.e", "prod.p", "prod.e",
                 "c", "c-x", "c-(c-x)", "bb", "a-(s-bb)", "b-bb"];
    println!("--- {name} ---");
    for i in 0..12 {
        let flag = if i == 5 { ' ' } else if g[i].to_bits() == want[i].to_bits() { ' ' } else { '*' };
        println!("{flag} {:<10} got {:>24.17e}  want {:>24.17e}", names[i], g[i], want[i]);
    }
}

fn main() {
    #[cfg(feature = "cpu")]
    run::<cubecl::cpu::CpuRuntime>("cpu");
    #[cfg(feature = "wgpu")]
    run::<cubecl::wgpu::WgpuRuntime>("wgpu");
}
