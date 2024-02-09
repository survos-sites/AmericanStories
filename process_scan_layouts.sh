python ./AmericanStories/src/run_img2layout_pipeline.py --manifest_path ./scans \
									    --checkpoint_path_layout ./american_stories_models/layout_model_new.onnx \
									    --checkpoint_path_line ./american_stories_models/line_model_new.onnx \
									    --label_map_path_layout ./AmericanStories/src/label_maps/label_map_layout.json \
									    --label_map_path_line ./AmericanStories/src/label_maps/label_map_line.json \
								        --output_save_path ./AmericanStories/outputs \
									    --layout_model_backend yolov8 \