use ndarray::concatenate;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::Zip;
use numpy::*;
use pyo3::prelude::*;
use std::time::Instant;

/// computes the squared norms of the given two dimensional array
fn compute_sq_norms(data: &ArrayView<f64, Ix2>) -> Array<f64, Ix1> {
    data.rows().into_iter().map(|row| row.dot(&row)).collect()
}

fn eucl(a: &ArrayView1<f64>, b: &ArrayView1<f64>, sq_norm_a: f64, sq_norm_b: f64) -> f64 {
    (sq_norm_a + sq_norm_b - 2.0 * a.dot(b)).sqrt()
}

fn argmax(v: &Array1<f64>) -> usize {
    let mut i = 0;
    let mut m = v[i];
    for idx in 1..v.shape()[0] {
        if v[idx] > m {
            i = idx;
            m = v[idx];
        }
    }
    i
}

fn greedy_minimum_maximum(data: &ArrayView2<f64>, k: usize) -> (Array1<usize>, Array1<usize>) {
    let n = data.shape()[0];
    let sq_norms = compute_sq_norms(&data);

    let first_center = 0usize;
    let mut centers: Array1<usize> = Array1::zeros(k);
    centers[0] = first_center;
    let mut distances: Array1<f64> = Array1::from_elem(n, f64::INFINITY);
    let mut assignment = Array1::<usize>::zeros(n);

    for i in 0..n {
        distances[i] = eucl(
            &data.row(first_center),
            &data.row(i),
            sq_norms[first_center],
            sq_norms[i],
        );
    }

    for idx in 1..k {
        let farthest = argmax(&distances);
        centers[idx] = farthest;
        for i in 0..n {
            let d = eucl(
                &data.row(farthest),
                &data.row(i),
                sq_norms[farthest],
                sq_norms[i],
            );
            if d < distances[i] {
                assignment[i] = idx;
                distances[i] = d;
            }
        }
    }

    (centers, assignment)
}

struct Coreset {
    /// The indices of the points in the original dataset
    point_ids: Array1<usize>,
    /// The actual points comprising the coreset
    points: Array2<f64>,
    /// The proxy function of the original point to the index (into point_ids) of their
    /// representative
    proxy: Array1<usize>,
    /// The number of points represented, for each proxy and color combination
    weights: Array2<u64>,
}

impl Coreset {
    fn new(
        data: &ArrayView2<f64>,
        tau: usize,
        colors: &ArrayView1<i64>,
        offset: Option<usize>,
    ) -> Self {
        let start = Instant::now();
        let offset = offset.unwrap_or(0);
        let (point_ids, proxy) = greedy_minimum_maximum(data, tau);
        let ncolors = colors.iter().max().unwrap() + 1;
        let mut points = Array2::<f64>::zeros((tau, data.ncols()));
        for (idx, point_id) in point_ids.iter().enumerate() {
            points.slice_mut(s![idx, ..]).assign(&data.row(*point_id));
        }
        let point_ids = point_ids + offset; // offset the IDs
        let mut weights = Array2::<u64>::zeros((tau, ncolors as usize));
        println!(
            "Colors shape {:?}, proxy shape {:?}",
            colors.shape(),
            proxy.shape()
        );
        Zip::from(colors)
            .and(&proxy)
            .for_each(|&color, &proxy_idx| {
                weights[[proxy_idx, color as usize]] += 1;
            });

        println!("Coreset built in {:?}", Instant::now() - start);

        Self {
            point_ids,
            points,
            proxy,
            weights,
        }
    }

    fn compose(&self, other: &Self) -> Self {
        println!(
            "Composing two coresets with {} and {} points",
            self.point_ids.len(),
            other.point_ids.len()
        );
        let point_ids =
            concatenate(Axis(0), &[self.point_ids.view(), other.point_ids.view()]).unwrap();
        let points = concatenate(Axis(0), &[self.points.view(), other.points.view()]).unwrap();
        let proxy = concatenate(
            Axis(0),
            &[
                self.proxy.view(),
                other.proxy.mapv(|p| p + self.point_ids.len()).view(),
            ],
        )
        .unwrap();
        let weights = concatenate(Axis(0), &[self.weights.view(), other.weights.view()]).unwrap();

        Self {
            point_ids,
            points,
            proxy,
            weights,
        }
    }

    fn new_parallel(
        processors: usize,
        data: &ArrayView2<f64>,
        tau: usize,
        colors: &ArrayView1<i64>,
    ) -> Self {
        let chunk_size = (data.nrows() + 1) / processors;
        data.axis_chunks_iter(Axis(0), chunk_size)
            .into_par_iter()
            .zip(colors.axis_chunks_iter(Axis(0), chunk_size))
            .enumerate()
            .map(|(chunk_idx, (chunk, chunk_colors))| {
                let off = chunk_size * chunk_idx;
                println!(
                    "chunk shape {:?} colors shape {:?} offset = {}",
                    chunk.shape(),
                    chunk_colors.shape(),
                    off
                );
                Coreset::new(&chunk.view(), tau, &chunk_colors.view(), Some(off))
            })
            .reduce_with(|c1, c2| c1.compose(&c2))
            .unwrap()
    }

    fn into_pytuple<'py>(
        self,
        py: Python<'py>,
    ) -> PyResult<(
        &'py PyArray1<usize>,
        &'py PyArray2<f64>,
        &'py PyArray1<usize>,
        &'py PyArray2<u64>,
    )> {
        Ok((
            self.point_ids.into_pyarray(py),
            self.points.into_pyarray(py),
            self.proxy.into_pyarray(py),
            self.weights.into_pyarray(py),
        ))
    }
}

#[pymodule]
fn fairkcenter(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "greedy_minimum_maximum", signature=(data, k, seed=1234))]
    fn gmm_py<'py>(
        py: Python<'py>,
        data: PyReadonlyArray2<f64>,
        k: usize,
        seed: u64,
    ) -> PyResult<(&'py PyArray1<usize>, &'py PyArray1<usize>)> {
        let start = Instant::now();
        let (centers, assignment) = greedy_minimum_maximum(&data.as_array(), k);
        println!("GMM took {:?}", Instant::now() - start);
        Ok((centers.into_pyarray(py), assignment.into_pyarray(py)))
    }

    #[pyfn(m)]
    fn build_coreset<'py>(
        py: Python<'py>,
        data: PyReadonlyArray2<f64>,
        tau: usize,
        colors: PyReadonlyArray1<i64>,
    ) -> PyResult<(
        &'py PyArray1<usize>,
        &'py PyArray2<f64>,
        &'py PyArray1<usize>,
        &'py PyArray2<u64>,
    )> {
        let coreset = Coreset::new(&data.as_array(), tau, &colors.as_array(), None);
        coreset.into_pytuple(py)
    }

    #[pyfn(m)]
    fn parallel_build_coreset<'py>(
        py: Python<'py>,
        processors: usize,
        data: PyReadonlyArray2<f64>,
        tau: usize,
        colors: PyReadonlyArray1<i64>,
    ) -> PyResult<(
        &'py PyArray1<usize>,
        &'py PyArray2<f64>,
        &'py PyArray1<usize>,
        &'py PyArray2<u64>,
    )> {
        eprintln!("data shape {:?}", data.shape());
        let coreset = Coreset::new_parallel(processors, &data.as_array(), tau, &colors.as_array());
        coreset.into_pytuple(py)
    }

    Ok(())
}
