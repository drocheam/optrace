
.. _dev_notes:

Notes
-----------

.. role:: python(code)
  :language: python
  :class: highlight

**Library Design**

* modular design with multiple submodules
* plotting and TraceGUI are independent of the tracing backend
* automation/scripting easily possible
* platform independent and open source
* importing only the backend tracing functionality does not import GUI and plotting libraries. This ensures better performance and enables us to use this functionality without ever needing to install this heavy libraries when they are never needed.

**Dependencies**

* balance of minimizing dependencies while also trying not to implement too much manually
* outsource GUI and plotting functionality/backend to external libraries
* tracing, color conversion, geometry manipulations etc. are implemented in library itself
* prefer python standard libraries
* only use open source dependencies

**Documentation Style**

* publication-like design
* LaTeX equations (🠪 `MathJax <https://www.mathjax.org/>`_)
* citations and bibliography (ideally per webpage)
* clickable references to sections, table, equations, citations
* keyword/topic index
* no use of external images (so no copyright/licensing issues)

**Graphical Outputs**

* professional, publication-like plots
* prefer serif fonts (e.g. Times New Roman)
* use LaTeX formatted equations/labels
* highlight more important elements by colors, font size or boldness
* axes, minor and major grid lines, titles and legend are a must
* :func:`block <optrace.plots.misc_plots.block>` function allows for halting the rest of the program when plotting 
* options for hiding some visual elements (labels, axes in GUI, legend entries)
* prefer svg for self-created figures, png for screenshots and webp for rendered images
* use tinypng to shrink pngs for the webpage, vecta.io for pyplot svgs
* in some cases there are issues with webp output in chromatic channels, export in lossless webp in these cases

**Multithreading and Multiprocessing**

* outsource heavy tasks onto multiple cores
* prefer threads over processes (no copying overhead). However we need to:
   * ensure there are no race conditions by using locks or by ensuring each thread writes on a different memory location
   * minimize python code (because of the GIL limitation while multithreading). Use lower level libraries with like numpy

* all worker and background threads need to terminate immediately when the main thread has ended
  This is ensured by providing :python:`daemon=True` while creating the :class:`Thread <threading.Thread>` object.
* multithreading should be controlled with a :attr:`multithreading <optrace.global_options.ClassGlobalOptions.multithreading>` global option
* threading rules for :class:`TraceGUI <optrace.gui.trace_gui.TraceGUI>`, with some ideas from `traits Futures <https://traits-futures.readthedocs.io/en/latest/guide/threading.html>`_
   * GUI changes only in main thread
   * GUI must be responsive at all costs
   * simplified Read/Write access model: only one thread can read and write
   * no locks in the main thread, waiting for locks only in worker threads

**Performance and Memory**

* use of multithreading
* :mod:`numpy` where possible
* avoid loops
* work with object references when possible (instead of copies)
* use masks on arrays instead of creating new ones
* only do calculations on elements actually needed
* pre-allocation instead of growing an array step by step
* use :python:`np.float32` or :python:`int` instead of :python:`np.float64` where high precision is not needed
* prefer analytical solutions instead of iterative numerical approximations
* multi-dimensional array access can be accelerated by choosing a specific memory layout of the 
  :class:`numpy.ndarray` (:python:`order="F", order="C"` etc.)
* always keep the GUI responsive
* range indexing (:python:`array[52:79829]`) is faster than boolean indexing  (:python:`array[[False, False, True, ..., True, False]]`,  which is faster than indexing by index list (:python:`array[[52, 456, 897987, 0, 77, ...]]`)
* a reasonable amount of points and curves in plots
* limit array sizes

**File Input/Output**

* OS independent file names
* store under a fallback path when files exists or can't be written
* write file in one go, don't append iteratively
* use external libraries for loading special files (images, spreadsheets etc.)
* use `compressed numpy archives <https://numpy.org/doc/stable/reference/generated/numpy.savez.html>`_ (``.npz``) to save some space
* try to detect the correct text encoding before loading any clear text formats. Outsource this to `chardet <https://github.com/chardet/chardet>`_

**Type and Value Checking**

* functions and classes that are exposed to the user should have type and value checking
* some people will tell you "this is not the pythonic way", however:
   * clear error messages are more helpful than needing to debug for 30 minutes
   * some values produce results that are valid mathematically, but impossible according to physics (e.g. negative energies, zero sized geometries etc.). 
   * even I as developer do not remember the types and value ranges for all parameters and the correct function usage

**Object and Geometry Locks**

* Class functions change parameters of the object itself, assigning parameters directly to the object can break things. Hard to debug, very frustrating.
   * e.g. the surface height depends on the surface parameters. Changing one without the other leads to very weird issues
   * This however depends on the parameters/variables and their roles. 
   * Some parameters are even read-only.
   * a change is not propagated to the parent/child object
   * we can't expect the user to know where assignments are possible without side effects and where not.
* ➜ restricting assignments
   * lock the geometry of an :python:`Element`, surfaces can only be assigned by special functions, 
     the position can only be changed with a dedicated method
   * lock objects like a :python:`Surface` to avoid nasty side effects
   * allow assignments where possible
   * make numpy arrays read-only while locking
   * locking can still be turned off manually when knowing about the internal mechanism, 
     but at that point the user should have noticed that changing the code in such a way was not intended by the library

**Coding Style**

* ``CamelCase`` class names, ``lower_case_with_underscores`` for functions, filenames, parameters. 
  Note that the latter this is not always possible for mathematical or physical quantities with standardized symbols
* increased line length of 120 characters
* prefer writing out keyword arguments (:python:`function(width=1026, height=4596)`)
  for readability and simpler documentation
* use of docstrings for functions, classes and important variables

**Standard Output**

* functions and classes output information and warnings to the terminal
* warnings have an own type :class:`OptraceWarning <optrace.warnings.OptraceWarning>` 
  and can be silenced with a global option 
  :attr:`show_warnings <optrace.global_options.ClassGlobalOptions.show_warnings>`
* the progressbar should only used for more time-intensive tasks and can be turned off with the :
  attr:`show_progressbar <optrace.global_options.ClassGlobalOptions.show_progress_bar>` option.

**Responsiveness**

* some actions need time, but the program should still appear active and responsive. Hence, we need to notify the user with:
   * a message that something has been started / processed/ approved
   * a progress bar indicating the progress and estimated remaining time. 
     Also distracts the user, lowering the subjective waiting time
* TraceGUI actions like tracing, focussing etc. need to run in background threads, 
  so the scene and main UI thread are responsive


