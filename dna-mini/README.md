# DNA Brain Mini

A small version of [`dna-neural-ca`](../dna-neural-ca) and [`dna-walker`](../dna-walker), written to be read top to
bottom. Open `index.html` in a browser and press **Start evolving**.

The script is in eight numbered steps:

1. **Read the DNA into genes.** A gene runs from `ATG` to a stop codon; each codon becomes a number 0–63.
2. **Turn each gene into rules.** Nine codons set one neuron type's rules: divisions, direction, daughter type,
   reach, which types it connects to, three weights, and its spiking threshold.
3. **Grow the brain on a grid.** One cell in the middle; every step, each cell with divisions left places a daughter
   in a neighbouring cell. This is the cellular automaton.
4. **Wire it up.** Each neuron connects to everything within its reach: neurons of the types it targets, the two
   sensors (food left, food right) and the two motors (go left, go right).
5. **Run the spiking brain.** Each neuron leaks, adds its input, and spikes above its threshold; spikes arrive on the
   next tick.
6. **The world and the score.** A creature on a line moves toward food. The score is food eaten plus a little
   credit for moving toward food, so near-misses beat doing nothing.
7. **Evolution.** 60 DNA strings per generation; the best 8 survive, the rest are mutated copies of them.
8. **Drawing and buttons.**

With these settings, 10 out of 10 test runs learned to chase food within 60 generations (median: generation 17).

Things to try: change a rule's range in step 2, put the sensors and motors further apart in step 3 (and watch
evolution get stuck), remove the credit for moving toward food in step 6, or make the world 2D with four sensors
and four motors.
