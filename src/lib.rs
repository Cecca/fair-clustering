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

/// returns the distance and the index in `set_indices` of the first point within `threshold`
fn set_eucl_threshold(
    data: &ArrayView2<f64>,
    sq_norms: &Array1<f64>,
    set_indices: &[usize],
    v: &ArrayView1<f64>,
    v_sq_norm: f64,
    threshold: f64,
) -> Option<(f64, usize)> {
    for (idx, i) in set_indices.iter().enumerate() {
        let d = eucl(v, &data.row(*i), v_sq_norm, sq_norms[*i]);
        if d <= threshold {
            return Some((d, idx));
        }
    }
    None
}

fn find_radius_range(data: &ArrayView2<f64>, k: usize) -> (f64, f64) {
    let n = data.shape()[0];
    let sq_norms = compute_sq_norms(&data);
    let mut diam = 0.0f64;
    for i in 0..n {
        let d = eucl(&data.row(0), &data.row(i), sq_norms[0], sq_norms[i]);
        diam = diam.max(d);
    }

    let mut minradius = std::f64::INFINITY;
    for i in 0..(k + 1) {
        for j in (i + 1)..k {
            let d = eucl(&data.row(i), &data.row(j), sq_norms[i], sq_norms[j]);
            minradius = minradius.min(d);
        }
    }
    assert!(minradius > 0.0);

    (minradius, diam)
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
    if n <= k {
        // Each point is its own center
        let centers = Array1::<usize>::from_iter(0..n);
        let assignment = Array1::<usize>::from_iter(0..n);
        return (centers, assignment);
    }

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
        ncolors: usize,
        offset: Option<usize>,
    ) -> Self {
        let offset = offset.unwrap_or(0);
        let start = Instant::now();
        let (point_ids, proxy) = greedy_minimum_maximum(data, tau);
        let mut points = Array2::<f64>::zeros((point_ids.len(), data.ncols()));
        for (idx, point_id) in point_ids.iter().enumerate() {
            points.slice_mut(s![idx, ..]).assign(&data.row(*point_id));
        }
        let point_ids = point_ids + offset; // offset the IDs
        let mut weights = Array2::<u64>::zeros((point_ids.len(), ncolors as usize));
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

    fn new_streaming(
        data: &ArrayView2<f64>,
        guess_step: f64,
        k: usize,
        tau: usize,
        colors: &ArrayView1<i64>,
        ncolors: usize,
    ) -> Self {
        let start = Instant::now();
        dbg!(k);
        // estimate upper and lower bound of the radius in the dataset
        let (lower, upper) = find_radius_range(data, k);

        let sq_norms = compute_sq_norms(data);

        // set up a log number of instances
        let mut guess = lower;
        while guess <= upper {
            println!("guess {} tau {}", guess, tau);
            // for each instance, go through the data points, and assign them
            let mut point_ids = Vec::new();
            let mut weights = Array2::<u64>::zeros((tau, ncolors));
            let mut proxy = Array1::zeros(data.nrows());
            let mut fail = false;

            for i in 0..data.nrows() {
                if let Some((_, earliest)) = set_eucl_threshold(
                    &data,
                    &sq_norms,
                    &point_ids,
                    &data.row(i),
                    sq_norms[i],
                    2.0 * guess,
                ) {
                    weights[(earliest, colors[i] as usize)] += 1;
                    proxy[i] = earliest;
                } else {
                    point_ids.push(i);
                    if point_ids.len() > tau {
                        fail = true;
                        break;
                    }
                    let earliest = point_ids.len() - 1; // the earliest is itself
                    proxy[i] = earliest;
                    weights[(earliest, colors[i] as usize)] = 1;
                }
            }

            if !fail {

                let point_ids = Array1::from_vec(point_ids);
                let mut points = Array2::<f64>::zeros((point_ids.len(), data.ncols()));
                for (idx, point_id) in point_ids.iter().enumerate() {
                    points.slice_mut(s![idx, ..]).assign(&data.row(*point_id));
                }
                let weights = weights.slice(s![0..point_ids.shape()[0], ..]).to_owned();
                assert_eq!(weights.sum(), data.nrows() as u64);
                eprintln!("Streaming completed in {:?}", start.elapsed());
                return Self {
                    point_ids,
                    points,
                    proxy,
                    weights,
                };
            }
            guess *= guess_step;
        }

        unreachable!()
    }

    fn compose(&self, other: &Self) -> Self {
        println!(
            "Composing two coresets with {} and {} points",
            self.point_ids.len(),
            other.point_ids.len()
        );
        let point_ids = concatenate(Axis(0), &[self.point_ids.view(), other.point_ids.view()])
            .expect("error composing point_ids");
        let points = concatenate(Axis(0), &[self.points.view(), other.points.view()])
            .expect("error composing points");
        assert_eq!(points.ncols(), self.points.ncols());
        assert_eq!(points.ncols(), other.points.ncols());
        let proxy = concatenate(
            Axis(0),
            &[
                self.proxy.view(),
                other.proxy.mapv(|p| p + self.point_ids.len()).view(),
            ],
        )
        .expect("error composing proxies");
        assert_eq!(self.weights.ncols(), other.weights.ncols());
        let weights = concatenate(Axis(0), &[self.weights.view(), other.weights.view()])
            .expect("error composing weights");

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
        ncolors: usize,
    ) -> Self {
        let chunk_size = ((data.nrows() as f64) / (processors as f64)).ceil() as usize;
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
                Coreset::new(&chunk.view(), tau, &chunk_colors.view(), ncolors, Some(off))
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

fn build_assignment(
    colors: &ArrayView1<i64>,
    proxy: &ArrayView1<usize>,
    coreset_ids: &ArrayView1<usize>,
    coreset_centers: &ArrayView1<usize>,
    coreset_assignment: &mut ArrayViewMut3<i64>,
) -> (Array1<usize>, Array1<usize>) {
    let shp = coreset_assignment.shape();
    let n = colors.len();
    let k = shp[1];

    let centers = coreset_centers.mapv(|c| coreset_ids[c]);
    let mut assignment = Array1::<usize>::ones(n) * 9999999;

    for x in 0..n {
        let p = proxy[x];
        let mut weight_to_distribute = coreset_assignment.slice_mut(s![p, .., colors[x] as usize]);
        for c in 0..k {
            if weight_to_distribute[c] > 0 {
                assignment[x] = c;
                weight_to_distribute[c] -= 1;
                break;
            }
        }
        assert!(assignment[x] < k, "point {}/{} was not assigned!", x, n);
    }

    println!("{:?}", coreset_assignment);

    assert!(
        *assignment.iter().max().unwrap() < k,
        "There are some unassigned points"
    );

    (centers, assignment)
}

#[pymodule]
fn fairkcenter(_py: Python, m: &PyModule) -> PyResult<()> {
    debug_assert!(false, "We should only run in release mode!");

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
        let colors = colors.as_array();
        let ncolors: usize = *colors.iter().max().unwrap() as usize + 1;
        let coreset = Coreset::new(&data.as_array(), tau, &colors, ncolors, None);
        coreset.into_pytuple(py)
    }

    #[pyfn(m)]
    fn streaming_build_coreset<'py>(
        py: Python<'py>,
        data: PyReadonlyArray2<f64>,
        guess_step: f64,
        k: usize,
        tau: usize,
        colors: PyReadonlyArray1<i64>,
    ) -> PyResult<(
        &'py PyArray1<usize>,
        &'py PyArray2<f64>,
        &'py PyArray1<usize>,
        &'py PyArray2<u64>,
    )> {
        let colors = colors.as_array();
        let ncolors: usize = *colors.iter().max().unwrap() as usize + 1;
        let coreset =
            Coreset::new_streaming(&data.as_array(), guess_step, k, tau, &colors, ncolors);
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
        let colors = colors.as_array();
        let ncolors: usize = *colors.iter().max().unwrap() as usize + 1;
        let coreset = Coreset::new_parallel(processors, &data.as_array(), tau, &colors, ncolors);
        coreset.into_pytuple(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "build_assignment")]
    fn build_assignment_py<'py>(
        py: Python<'py>,
        colors: PyReadonlyArray1<i64>,
        proxy: PyReadonlyArray1<usize>,
        coreset_ids: PyReadonlyArray1<usize>,
        coreset_centers: PyReadonlyArray1<usize>,
        mut coreset_assignment: PyReadwriteArray3<i64>,
    ) -> PyResult<(&'py PyArray1<usize>, &'py PyArray1<usize>)> {
        let mut coreset_assignment = coreset_assignment.as_array_mut();
        let start = Instant::now();
        let (centers, assignment) = build_assignment(
            &colors.as_array(),
            &proxy.as_array(),
            &coreset_ids.as_array(),
            &coreset_centers.as_array(),
            &mut coreset_assignment,
        );
        eprintln!("Assignment of original points {:?}", Instant::now() - start);
        Ok((centers.into_pyarray(py), assignment.into_pyarray(py)))
    }

    Ok(())
}
