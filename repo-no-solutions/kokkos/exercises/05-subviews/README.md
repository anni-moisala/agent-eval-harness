<!--
SPDX-FileCopyrightText: 2024 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Subviews

In the exercise you can play with subviews. The skeleton code `subviews.cpp` contains
a single 2D array. Your tasks are following:

1. Create subviews for the boundaries (top,  bottom, left, right), and initialize
the boundaries with the hel pof subviews using a `parallel_for`: set every element
of the top boundary to `-1`, bottom to `-2`, left to `-5`, and right to `-6`.
2. Copy the boundary data to the host. For this, you need contiguous buffers on the device (copying data between non-contiguous views is possible only within the same execution space), and their mirror views on the host.
3. Print out the boundary values on the host: for each boundary, sum its values --
excluding its two corner elements, which are shared with an adjacent boundary --
and print one line as `<label> <sum>`, using exactly these labels:

   ```
   Top boundary <sum>
   Bottom boundary <sum>
   Left boundary <sum>
   Right boundary <sum>
   ```
