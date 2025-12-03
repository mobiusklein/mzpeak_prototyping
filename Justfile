export RUST_LOG := "trace"

small:
    cargo r -r --example convert -- -y -z -u small.mzML -o small.mzpeak
    cargo r -r --example convert -- -p -c -y -z -u small.mzML -o small.chunked.mzpeak

small_chunked:
    cargo r --example convert -- -p -c -y -z -u small.mzML -o small.chunked.mzpeak

test:
    cargo t --tests -- --no-capture

alias t := test