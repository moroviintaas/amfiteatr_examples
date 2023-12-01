use std::cmp::Ordering;
use std::path::Path;
use plotters::prelude::*;

pub fn plot_payoffs(file: &Path, payoffs: &[f32]) -> Result<(), Box<dyn std::error::Error>>{
    let root  = SVGBackend::new(&file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let min = match payoffs.iter().min_by(|a, b |{
        a.partial_cmp(b).unwrap_or(Ordering::Equal)
    }){
        None => 0.0,
        Some(n) if n < &0.0 => *n,
        Some(_) => 0.0f32
    };

    let max = match payoffs.iter().max_by(|a, b |{
        a.partial_cmp(b).unwrap_or(Ordering::Equal)
    }){
        None => 0.0,
        Some(n) if n > &0.0 => *n,
        Some(_) => 0.0f32
    };

    let mut chart = ChartBuilder::on(&root)
        .caption("payoffs", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..payoffs.len() as f32, min..max)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            (0..payoffs.len()).map(|x| (x as f32, payoffs[x])),
            &RED,
        ))?
        .label("payoffs")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

pub fn plot_2payoffs(file: &Path, payoffs_0: &[f32], payoffs_1: &[f32]) -> Result<(), Box<dyn std::error::Error>>{
    let root  = SVGBackend::new(&file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let min = match payoffs_0.iter().chain(payoffs_1.iter()).min_by(|a, b |{
        a.partial_cmp(b).unwrap_or(Ordering::Equal)
    }){
        None => 0.0,
        Some(n) if n < &0.0 => *n,
        Some(_) => 0.0f32
    };


    let max = match payoffs_0.iter().chain(payoffs_1.iter()).max_by(|a, b |{
        a.partial_cmp(b).unwrap_or(Ordering::Equal)
    }){
        None => 0.0,
        Some(n) if n > &0.0 => *n,
        Some(_) => 0.0f32
    };


    let mut chart = ChartBuilder::on(&root)
        .caption("payoffs", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..payoffs_0.len() as f32, min..max)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            (0..payoffs_0.len()).map(|x| (x as f32, payoffs_0[x])),
            &RED,
        ))?
        .label("agent 0")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart

        .draw_series(LineSeries::new(
            (0..payoffs_1.len()).map(|x| (x as f32, payoffs_1[x])),
            &BLUE,
        ))?
        .label("agent 1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

