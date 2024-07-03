use nalgebra::{Const, OMatrix, SVector};

pub struct Halfspace<const N: usize> {
    b: SVector<f64, N>,
    n: SVector<f64, N>,
}

pub struct Span<const N: usize> {
    u: SVector<f64, N>,
    v: SVector<f64, N>,
}

pub struct Domain<const N: usize> {
    low: SVector<f64, N>,
    high: SVector<f64, N>,
}

impl<const N: usize> Span<N> {
    // Provides a span, composed of two vectors which will be orthonormalized.
    fn new(u: SVector<f64, N>, v: SVector<f64, N>) -> Self {
        let u = u.normalize();
        let v = v.normalize();
        let v = (v - u * u.dot(&v)).normalize();
        return Span { u, v };
    }

    // Provides a rotater function rot(angle: f64) which returns a rotation matrix
    // that rotates by angle radians along the @&self span.
    fn get_rotater(&self) -> impl Fn(f64) -> OMatrix<f64, Const<N>, Const<N>> {
        let identity = OMatrix::<f64, Const<N>, Const<N>>::identity();

        let a = self.v * self.v.transpose() - self.u * self.v.transpose();
        let b = self.u * self.u.transpose() - self.v * self.v.transpose();

        move |angle: f64| identity + a * angle.sin() + b * (angle.cos() - 1.0)
    }
}

impl<const N: usize> Domain<N> {
    // fn normalized() -> Self {
    //     let low = SVector::<f64, N>::zeros();
    //     let high = SVector::<f64, N>::repeat(1.0);
    //     return Domain { low, high };
    // }

    fn dimensions(&self) -> SVector<f64, N> {
        return self.high - self.low;
    }

    fn translate_point_domains(
        p: SVector<f64, N>,
        from: Domain<N>,
        to: Domain<N>,
    ) -> SVector<f64, N> {
        ((p - from.low).component_div(&from.dimensions())).component_mul(&to.dimensions()) + to.low
    }
}
