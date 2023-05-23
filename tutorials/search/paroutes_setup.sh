# Set up directory to store files (with gitignore)
files_dir="paroutes_files"
mkdir -p "$files_dir"
echo "*" > "${files_dir}/.gitignore"

# Clone repo
cd "$files_dir"
git clone https://github.com/MolecularAI/PaRoutes.git
cd PaRoutes

# Download the data
python data/download_data.py

# Move stuff
for suffix in "hdf5" "csv" "txt" "json" "csv.gz" ; do
    echo "Moving .${suffix} files..."
    mv data/*".${suffix}" ..
done

# Clean up the PaRoutes repo
cd ..
rm -rf PaRoutes
