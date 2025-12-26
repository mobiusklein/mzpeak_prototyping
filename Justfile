export RUST_LOG := "debug"

small:
    cargo r -r --example convert -- -y -z -u small.mzML -o small.mzpeak
    cargo r -r --example convert -- -p -c -y -z -u small.mzML -o small.chunked.mzpeak

small_chunked:
    cargo r --example convert -- -p -c -y -z -u small.mzML -o small.chunked.mzpeak

test:
    cargo t --tests -- --no-capture

pytest:
    py.test -l -s -v python/test/ \
        --cov=mzpeak --cov-report term \
        --log-level=DEBUG --cov-report html

alias t := test