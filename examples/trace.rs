use cube_math::tables::double::exp as t;
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn trace(inp: &Array<f64>, out: &mut Array<f64>, tab: &Array<u64>) {
    if ABSOLUTE_POS < 1usize {
        let x = inp[0];
        let kd_s = fma(x, t::INVLN2N, t::SHIFT);
        let ki = u64::reinterpret(kd_s);
        let kd = kd_s - t::SHIFT;
        let inner = fma(kd, t::NEGLN2HIN, x);
        let r = fma(kd, t::NEGLN2LON, inner);
        let idx = usize::cast_from((ki & 127u64) * 2u64);
        let tail = f64::reinterpret(tab[idx]);
        let sbits = tab[idx + 1] + (ki << 45u64);
        let p12 = fma(r, t::C3, t::C2);
        let t3 = tail + r;
        let r2 = r * r;
        let p45 = fma(r, t::C5, t::C4);
        let s1 = fma(r2, p12, t3);
        let r4 = r2 * r2;
        let tmp = fma(r4, p45, s1);
        let scale = f64::reinterpret(sbits);
        out[0] = kd_s;
        out[1] = kd;
        out[2] = inner;
        out[3] = r;
        out[4] = tail;
        out[5] = f64::reinterpret(sbits);
        out[6] = p12;
        out[7] = s1;
        out[8] = tmp;
        out[9] = fma(scale, tmp, scale);
        out[10] = f64::cast_from(idx);
        out[11] = t::INVLN2N;
        out[12] = t::SHIFT;
        out[13] = t::C2;
        out[14] = t::NEGLN2LON;
        out[15] = fma(2.0f64, 3.0f64, 1.0f64);
    }
}

fn host(x: f64) -> Vec<f64> {
    let kd_s = x.mul_add(t::INVLN2N, t::SHIFT);
    let ki = kd_s.to_bits();
    let kd = kd_s - t::SHIFT;
    let inner = kd.mul_add(t::NEGLN2HIN, x);
    let r = kd.mul_add(t::NEGLN2LON, inner);
    let idx = ((ki & 127) * 2) as usize;
    let tail = f64::from_bits(t::TAB[idx]);
    let sbits = t::TAB[idx + 1].wrapping_add(ki << 45);
    let p12 = r.mul_add(t::C3, t::C2);
    let t3 = tail + r;
    let r2 = r * r;
    let p45 = r.mul_add(t::C5, t::C4);
    let s1 = r2.mul_add(p12, t3);
    let r4 = r2 * r2;
    let tmp = r4.mul_add(p45, s1);
    let scale = f64::from_bits(sbits);
    vec![
        kd_s, kd, inner, r, tail, scale, p12, s1, tmp,
        scale.mul_add(tmp, scale), idx as f64,
        t::INVLN2N, t::SHIFT, t::C2, t::NEGLN2LON, 2.0f64.mul_add(3.0, 1.0),
    ]
}

fn run<R: Runtime>(name: &str, x: f64) {
    let client = R::client(&Default::default());
    let inp = client.create(cubecl::bytes::Bytes::from_elems(vec![x]));
    let tabv = t::TAB.to_vec();
    let tl = tabv.len();
    let tabh = client.create(cubecl::bytes::Bytes::from_elems(tabv));
    let outh = client.empty(16 * 8);
    unsafe {
        trace::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(inp, 1),
            ArrayArg::from_raw_parts(outh.clone(), 16),
            ArrayArg::from_raw_parts(tabh, tl),
        );
    }
    let o = client.read_one(outh).unwrap();
    let g = f64::from_bytes(&o);
    let h = host(x);
    let names = ["kd_s", "kd", "inner", "r", "tail", "scale", "p12", "s1", "tmp", "result", "idx", "INVLN2N", "SHIFT", "C2", "NEGLN2LON", "fma(2,3,1)"];
    println!("--- {name}, x = {x} ---");
    for i in 0..16 {
        let flag = if g[i].to_bits() == h[i].to_bits() { " " } else { "*" };
        println!("{flag} {:<10} gpu {:>24.17e} ({:#018x})  host {:>24.17e} ({:#018x})", names[i], g[i], g[i].to_bits(), h[i], h[i].to_bits());
    }
}

fn main() {
    run::<cubecl::cpu::CpuRuntime>("cpu", 1.0);
}
