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

fn closest_all(
    data: &ArrayView2<f64>,
    sq_norms: &Array1<f64>,
    v: &ArrayView1<f64>,
    v_sq_norm: f64,
) -> (f64, usize) {
    let mut min_dist = std::f64::INFINITY;
    let mut min_idx = 0;
    for i in 0..data.nrows() {
        let d = eucl(v, &data.row(i), v_sq_norm, sq_norms[i]);
        if d <= min_dist {
            min_dist = d;
            min_idx = i;
        }
    }
    return (min_dist, min_idx);
}

fn closest(
    data: &ArrayView2<f64>,
    sq_norms: &Array1<f64>,
    set_indices: &[usize],
    v: &ArrayView1<f64>,
    v_sq_norm: f64,
) -> (f64, usize) {
    let mut min_dist = std::f64::INFINITY;
    let mut min_idx = 0;
    for i in set_indices.iter() {
        let d = eucl(v, &data.row(*i), v_sq_norm, sq_norms[*i]);
        if d <= min_dist {
            min_dist = d;
            min_idx = *i;
        }
    }
    return (min_dist, min_idx);
}

fn find_radius_range(data: &ArrayView2<f64>, k: usize) -> (f64, f64) {
    let n = data.shape()[0];
    let sq_norms = compute_sq_norms(&data);
    let mut diam = 0.0f64;
    for i in 0..n {
        let d = eucl(&data.row(0), &data.row(i), sq_norms[0], sq_norms[i]);
        diam = diam.max(d);
    }

    // get the first k plus one points, without duplicates
    let mut first_kp1 = Vec::new();
    first_kp1.push(0);
    let mut i = 0;
    while first_kp1.len() < k + 1 && i < data.nrows() {
        let (d, _) = closest(data, &sq_norms, &first_kp1, &data.row(i), sq_norms[i]);
        if d > 0.0 {
            first_kp1.push(i);
        }
        i += 1;
    }

    let mut minradius = std::f64::INFINITY;
    for i in 0..first_kp1.len() {
        let ii = first_kp1[i];
        for j in (i + 1)..first_kp1.len() {
            let jj = first_kp1[j];
            let d = eucl(&data.row(ii), &data.row(jj), sq_norms[ii], sq_norms[jj]);
            assert!(d > 0.0);
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

struct StreamingInstance {
    guess: f64,
    tau: usize,
    point_ids: Vec<usize>,
    weights: Array2<u64>,
    proxy: Array1<usize>,
    fail: bool,
}
impl StreamingInstance {
    fn size_bytes(&self) -> usize {
        use std::mem::size_of;
        size_of::<usize>() * (self.point_ids.len() + self.proxy.len())
            + size_of::<u64>() * self.weights.len()
    }
    fn new(guess: f64, data: &ArrayView2<f64>, tau: usize, ncolors: usize) -> Self {
        let point_ids = Vec::new();
        let weights = Array2::<u64>::zeros((tau, ncolors));
        let proxy = Array1::zeros(data.nrows());
        let fail = false;
        Self {
            guess,
            tau,
            point_ids,
            weights,
            proxy,
            fail,
        }
    }
    fn update(
        &mut self,
        data: &ArrayView2<f64>,
        sq_norms: &Array1<f64>,
        colors: &ArrayView1<i64>,
        i: usize,
    ) {
        if self.fail {
            return;
        }
        if let Some((_, earliest)) = set_eucl_threshold(
            &data,
            &sq_norms,
            &self.point_ids,
            &data.row(i),
            sq_norms[i],
            2.0 * self.guess,
        ) {
            self.weights[(earliest, colors[i] as usize)] += 1;
            self.proxy[i] = earliest;
        } else {
            self.point_ids.push(i);
            if self.point_ids.len() > self.tau {
                self.fail = true;
                return;
            }
            let earliest = self.point_ids.len() - 1; // the earliest is itself
            self.proxy[i] = earliest;
            self.weights[(earliest, colors[i] as usize)] = 1;
        }
    }
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
    ) -> (Self, usize) {
        let start = Instant::now();
        // estimate upper and lower bound of the radius in the dataset
        let (lower, upper) = find_radius_range(data, k);
        let mut instances = Vec::new();
        let mut guess = lower;
        while guess <= upper {
            instances.push(StreamingInstance::new(guess, data, tau, ncolors));
            guess *= guess_step;
        }
        eprintln!("There are {} streaming instances", instances.len());

        let sq_norms = compute_sq_norms(data);

        for i in 0..data.nrows() {
            for instance in instances.iter_mut() {
                instance.update(data, &sq_norms, colors, i);
            }
        }

        let total_size_bytes = instances
            .iter()
            .map(|inst| inst.size_bytes())
            .sum::<usize>();

        for mut instance in instances.into_iter() {
            if !instance.fail {
                if instance.point_ids.len() < k {
                    // add arbitrary points, with 0 weight
                    let mut i = data.nrows() - 1;
                    while instance.point_ids.len() < k {
                        if !instance.point_ids.contains(&i) {
                            instance.point_ids.push(i);
                        }
                        i -= 1;
                    }
                }
                let point_ids = Array1::from_vec(instance.point_ids);
                let mut points = Array2::<f64>::zeros((point_ids.len(), data.ncols()));
                for (idx, point_id) in point_ids.iter().enumerate() {
                    points.slice_mut(s![idx, ..]).assign(&data.row(*point_id));
                }
                let weights = instance
                    .weights
                    .slice(s![0..point_ids.shape()[0], ..])
                    .to_owned();
                assert_eq!(weights.sum(), data.nrows() as u64);
                eprintln!("Streaming completed in {:?}", start.elapsed());
                return (
                    Self {
                        point_ids,
                        points,
                        proxy: instance.proxy.clone(),
                        weights,
                    },
                    total_size_bytes,
                );
            }
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

    fn new_parallel_tiny(
        processors: usize,
        data: &ArrayView2<f64>,
        tau: usize,
        colors: &ArrayView1<i64>,
        ncolors: usize,
    ) -> Self {
        fn cat_rows<T: Clone, S: ndarray::Dimension + ndarray::RemoveAxis>(
            a: Array<T, S>,
            b: Array<T, S>,
        ) -> Array<T, S> {
            ndarray::concatenate(Axis(0), &[a.view(), b.view()]).unwrap()
        }

        let chunk_size = ((data.nrows() as f64) / (processors as f64)).ceil() as usize;
        let (center_ids, center_points): (Array1<usize>, Array2<f64>) = data
            .axis_chunks_iter(Axis(0), chunk_size)
            .into_par_iter()
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let off = chunk_size * chunk_idx;
                let (center_ids, _) = greedy_minimum_maximum(&chunk.view(), tau);
                let center_points = chunk.select(Axis(0), &center_ids.view().to_slice().unwrap());
                let center_ids = center_ids + off;
                (center_ids, center_points)
            })
            .reduce_with(|(ids1, points1), (ids2, points2)| {
                (cat_rows(ids1, ids2), cat_rows(points1, points2))
            })
            .unwrap();
        assert_eq!(data.ncols(), center_points.ncols());

        // build the "coreset of the coreset"
        let (selected_ids, _) = greedy_minimum_maximum(&center_points.view(), tau);
        let point_ids = center_ids.select(Axis(0), selected_ids.view().to_slice().unwrap());
        let points = center_points.select(Axis(0), selected_ids.view().to_slice().unwrap());
        let points_sq_norms = compute_sq_norms(&points.view());

        // now build, in parallel, the proxy and weight functions
        let (proxy, weights): (Array1<usize>, Array2<u64>) = data
            .axis_chunks_iter(Axis(0), chunk_size)
            .into_par_iter()
            .zip(colors.axis_chunks_iter(Axis(0), chunk_size))
            .map(|(chunk, chunk_colors)| {
                let sq_norms = compute_sq_norms(&chunk);
                let mut proxy = Array1::<usize>::zeros(chunk.nrows());
                let mut weights = Array2::<u64>::zeros((point_ids.len(), ncolors as usize));
                for i in 0..chunk.nrows() {
                    let (_, p) =
                        closest_all(&points.view(), &points_sq_norms, &chunk.row(i), sq_norms[i]);
                    proxy[i] = p;
                    weights[(p, chunk_colors[i] as usize)] += 1;
                }
                (proxy, weights)
            })
            .reduce_with(|(p1, w1), (p2, w2)| (cat_rows(p1, p2), w1 + w2))
            .unwrap();

        Self {
            point_ids,
            points,
            proxy,
            weights,
        }
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
        usize,
    )> {
        let colors = colors.as_array();
        let ncolors: usize = *colors.iter().max().unwrap() as usize + 1;
        let (coreset, total_size_bytes) =
            Coreset::new_streaming(&data.as_array(), guess_step, k, tau, &colors, ncolors);
        let (ids, points, proxy, weights) = coreset.into_pytuple(py)?;
        Ok((ids, points, proxy, weights, total_size_bytes))
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
    fn parallel_build_coreset_tiny<'py>(
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
        let coreset =
            Coreset::new_parallel_tiny(processors, &data.as_array(), tau, &colors, ncolors);
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
