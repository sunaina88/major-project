mkdir -p logs
run() { tag=$1; shift; echo "=== $tag ==="; python models/stgnn/annual.py --tag "$tag" "$@" > "logs/$tag.log" 2>&1 || echo "FAILED: $tag (see logs/$tag.log)"; tail -4 "logs/$tag.log"; }

run geo_w
run geo_now --weather 0
run identity --graph identity
run knn --graph knn
run dist --graph dist
run hybrid --graph hybrid
run gat --kind gat
run single --single 1
run K2 --K 2
run K5 --K 5
run horizons --horizons 1,2,3

echo; echo "################ FINAL COMPARISON ################"
python models/stgnn/compare_runs.py geo_w geo_now identity knn dist hybrid gat single K2 K5 horizons@2 horizons@3

run final --forecast 1 --horizons 1,2,3
python models/stgnn/predict.py