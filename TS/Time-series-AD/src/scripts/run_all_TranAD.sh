for script in scripts/TranAD/*.sh; do
    chmod +x "$script"     # Make the script executable if it's not already
    ./"$script"            # Execute the script
done