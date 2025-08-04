#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

BUCKET_NAME="gs://cmu-gpucloud-mengyan3"
dset_base_name="flan_plain"

#rm -rf "/compute/babel-14-29/mengyan3/${dset_base_name}"
#rm -rf "/compute/babel-14-29/mengyan3/${dset_base_name}_val"
#mkdir -p "/compute/babel-14-29/mengyan3/${dset_base_name}"
#mkdir -p "/compute/babel-14-29/mengyan3/${dset_base_name}_val"

#gcloud storage cp -r "${BUCKET_NAME}/${dset_base_name}" "/compute/babel-14-29/mengyan3/"

echo "Starting restructuring process..."
BASE="/compute/babel-14-29/mengyan3/${dset_base_name}"
VAL_DEST="/compute/babel-14-29/mengyan3/${dset_base_name}_val"

# Clean up any saved_chunk_* directories first
echo "Cleaning up saved_chunk_* directories..."
find "$BASE" -type d -name "saved_chunk_*" -exec rm -rf {} + 2>/dev/null || echo "No saved_chunk directories found"

# First, copy the val directory from the first shard
# if [ -d "$BASE/1/val" ]; then
#     echo "Copying validation data from $BASE/1/val to $VAL_DEST"
#     cp -r "$BASE/1/val/"* "$VAL_DEST/"
# else
#     echo "Error: No 'val' directory found in $BASE/1. Aborting."
#     exit 1
# fi

# Loop through each numbered folder to restructure
for i in {1..50}; do
    if [ -d "$BASE/$i" ]; then
        echo "Processing directory: $BASE/$i"
        
        # Clean up any saved_chunk_* directories in this shard
        echo "  Cleaning up saved_chunk_* directories in $BASE/$i"
        find "$BASE/$i" -type d -name "saved_chunk_*" -exec rm -rf {} + 2>/dev/null || echo "  No saved_chunk directories found in $BASE/$i"
        
        # Move train content up to the shard folder
        if [ -d "$BASE/$i/train" ]; then
            echo "  Moving files from $BASE/$i/train to $BASE/$i"
            # Use find to avoid "Argument list too long" errors with many files
            find "$BASE/$i/train" -type f -exec mv {} "$BASE/$i/" \;
            echo "  Removing directory $BASE/$i/train"
            rm -rf "$BASE/$i/train"
        else
            echo "  No 'train' directory found in $BASE/$i, skipping."
        fi
        
        # Remove val directory from each shard
        if [ -d "$BASE/$i/val" ]; then
            echo "  Removing 'val' directory from $BASE/$i"
            rm -rf "$BASE/$i/val"
        else
            echo "  No 'val' directory in $BASE/$i, skipping removal."
        fi
    else
        echo "Directory $BASE/$i does not exist, skipping."
    fi
done

echo "Restructuring process completed successfully."