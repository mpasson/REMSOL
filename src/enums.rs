#[derive(Copy, Clone)]
pub enum BackEnd {
    Scattering,
    Transfer,
}

#[derive(Copy, Clone)]
pub enum Polarization {
    TE,
    TM,
}
