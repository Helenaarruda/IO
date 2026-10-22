## Setup

Install [uvr](https://github.com/nbafrank/uvr) if you have not already.

In your terminal, navigate to the Part 2 project root (the folder
containing `uvr.toml`, `uvr.lock`, and the R scripts). Then run:

```bash
uvr sync
```

Make sure the provided data files are in `data/input/`.

## Run Part 2

Run these commands from the project root:

```bash
uvr run code/pset2_q2.R
uvr run code/pset2_q2_nested.R
uvr run code/pset_q2_nested_welfare.R
uvr run code/pset_q2_three_nests.R
```

The scripts run in sequence. The process stops if any script fails.