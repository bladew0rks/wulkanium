// stamp the build time so the log shows which build ran
fn main() {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    println!("cargo:rustc-env=BUILD_UNIX_TIME={secs}");
    println!("cargo:rerun-if-changed=src");
}
