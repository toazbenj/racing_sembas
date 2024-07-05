use nalgebra::SVector;

pub fn svector_to_array<const N: usize>(v: SVector<f64, N>) -> [f64; N] {
    // let mut arr = [0.0; N];
    // for i in 0..N {
    //     arr[i] = v[i];
    // }
    // return arr;

    let arr;
    unsafe {
        let ptr = v.as_ptr();
        arr = std::slice::from_raw_parts(ptr, N)
            .clone()
            .try_into()
            .unwrap();
    }

    return arr;
}

pub fn array_distance<const N: usize>(a1: &[f64; N], a2: &[f64; N]) -> f64 {
    let v1: SVector<f64, N> = unsafe {
        let a1_ptr = a1.as_ptr();
        SVector::from_column_slice(std::slice::from_raw_parts(a1_ptr, N))
    };

    let v2: SVector<f64, N> = unsafe {
        let a2_ptr = a2.as_ptr();
        SVector::from_column_slice(std::slice::from_raw_parts(a2_ptr, N))
    };

    return (v2 - v1).norm();
}
