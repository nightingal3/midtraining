#!/bin/bash
echo "Syncing back changes to the repo (not attempting to sync changes to conda)"

set -ex
scp -r /tmp/workspace-upperdir/* nightingal3@devgpu003.rva3.facebook.com:/data/users/nightingal3/all_in_one_pretraining/deploy
