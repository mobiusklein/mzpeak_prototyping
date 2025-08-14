export RUST_LOG := "debug"

small:
    cargo r -r --example convert -- -y -z -u small.mzML -o small.mzpeak
    cargo r -r --example convert -- -c -y -z -u small.mzML -o small.chunked.mzpeak

test:
    cargo t

alias t := test