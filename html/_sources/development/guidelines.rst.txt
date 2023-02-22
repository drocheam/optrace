
.. _guidelines:

General Guidelines
-----------------------


**Library Design**

* modular design with multiple submodules
* plotting and TraceGUI are independent of the tracing backend
* automation/scripting easily possible
* platform independent and open source
* importing only the backend tracing functionality does not import GUI and plotting libraries. 
  This ensures better performance and enables us to use this functionality without 
  ever needing to install this heavy libraries when they are never needed.


**Dependencies**

* balance of minimizing dependencies while also trying not to implemnt too much manually
* outsource GUI and plotting functionality/backend to external libraries
* tracing, color conversion, geometry manipulations etc. are implemented in library itself
* prefer python standard libraries
* use only open source dependencies


**Documentation Style**

* publication-like design
* serif fonts (Times New Roman, Cambria, ...)
* LaTeX equations (🠪 `MathJax <https://www.mathjax.org/>`_)
* citations and bibliography (ideally per webpage)
* clickable references to sections, table, equations, citations
* keyword/topic index
* no use or including of external images, all images should be self-made (so no copyright issues)


**Graphical Outputs**

* professional, publication-like plots
* prefer serif fonts (e.g. Times New Roman)
* minimize different fonts
* use LaTeX formatted equations/labels
* highlight more important elements by colors, font size or boldness
* axes, minor and major grid lines, titles and legend are a must
* ``block`` option allows for halting the rest of the program when plotting or continuing it with ``block=False``
* options for hiding some visual elements (labels, axes in GUI, legend entries)


**Multithreading and Multiprocessing**

* outsource heavy calculation tasks onto multiple cores
* prefer threads over processes (no copying overhead). However we need to:
   * ensure there are no race conditions by using locks or by ensuring each thread writes on a different memory location
   * minimize python code (because of the GIL limitation while multithreading). Use lower level libraries with no python code like numpy for heavy tasks

* all worker and background threads need to terminate immediately when the main thread is terminated. 
  This is ensured by providing ``daemon=True`` while creating the ``Thread`` object.
* all classes and functions using multithreading should have a boolean threading option that can be set to ``threading=False``. 
  Turning off threading is useful when debugging, profiling or running multiple things in parrallel.
* threading rules ``TraceGUI``, with some ideas from `traits Futures <https://traits-futures.readthedocs.io/en/latest/guide/threading.html>`_
   * GUI changes only in main thread
   * the GUI must be responsive at all costs
   * simplified Read/Write access model: only one thread can read and write
   * no locks in main thread, waiting for locks only in worker threads


**Performance and Memory**

* use multithreading
* ``numpy`` where possible
* avoid loops
* sometimes noticeable speed-ups when using ``numexpr`` over ``numpy``
* work with references when possible (instead copies)
* use masks on arrays instead of creating new ones
* only do calculations on elements actually needed
* initial initialization instead of growing an array step by step
* use ``float32`` or ``int`` instead of ``float64`` where high precision is not needed
* prefer analytical solutions instead of iterative numerical approximations
* multi-dimensional array access can profit from the correct memory layout of the ``numpy.ndarray`` (``order="F", order="C"`` etc.)
* always keep the GUI responsive
* range indexing (``array[52:79829]``) is faster than boolean indexing (``array[[False, False, True, ..., True, False]]``, 
  which is faster than indexing by index list (``array[[52, 456, 897987, 0, 77, ...]]``)
* limit array sizes
* store results that will be needed in the future or are accessed multiple times
* when plotting only use a reasonable amount of points and curves


**File Input/Output**

* OS independent file names
* save under a fallback path when files exists/ can't be written
* write file in one go, don't append iteratively
* use external library for loading special files (images, spreadsheets etc.)
* use `compressed numpy archives <https://numpy.org/doc/stable/reference/generated/numpy.savez.html>`_ (``.npz``) to save some space
* don't save anything in any proprietary formats
* try to detect the correct text encoding before loading any clear text formats. Outsource the work to `chardet <https://github.com/chardet/chardet>`_
* only load files once, no need to reload them


**Type and Value Checking**

* functions and classes that are used and exposed to the user should have type and value checking
* some people will tell you "this is not the pythonic way", however:
   * clear error messages before an action are more helpful than trying for 20 minutes to comprehend some internal exception
   * even the author does not remember the types and value ranges of all function parameters
   * some values produce results that are valid mathematically, but impossible according to physics (e.g. negative energies, zero sized geometries etc.). 
     This can sometimes appear unexpectedly, so value checking is crucial.


**Object and Geometry Locks**

* Class functions change parameters of the object itself, assigning parameters directly to the object can break things. Hard to debug, very frustrating.
   * e.g. the surface height depends on the surface parameters. Changing one without the other leads to very weird issues
   * This however depends on the parameters/variables and their roles. 
   * Some parameters are even read-only.
   * a change is not propagated to the parent/child object
   * we can't expect the user to know where assignments are possible without side effects and where not. The library has too much complexity for this.
* ➜ restricting assignments
   * ignore people saying "this is not pythonic!!"
   * lock the geometry of an ``Element``, surfaces can only be assigned by using a special functions, the position can only be changed with a dedicated method
   * lock objects like a ``Surface`` to avoid nasty side effects
   * allow assignments where possible
   * make numpy arrays read-only while locking
   * when knowing the internals of the locking mechanism the user could turn it off. Let's hope warning and info messages convince him not to do so.


**Comments**

* use of docstrings for functions, classes and important variables
* Comments should do the following things
   * explain what a function does and what parameters and return values are
   * describe what a module of class does
   * describe steps inside the function
   * provide additional information why something is done
   * subdivide a longer function into smaller parts
   * link to documentation resources or related sources


**Coding Style**

* ``CamelCase`` class names, ``lower_case_with_underscores`` for functions, filenames, parameters. Note that the latter this is not always possible when mathematical or physical quantities are used
* increased line length of 120 characters
* prefer writing out keyword arguments (``function(width=1026, height=4596)``) instead just the value (``function(1026, 4596)``) for readability and simpler documentation


**Standard Output**

* functions and classes output information and warnings, when operations are experimental, have inprecise results or use edge cases
* all classes and functions writing to standard output have a "silent" boolean parameter that can be set to ``False`` to mute output
* clear, explaining warning messages and exceptions


**Responsiveness**

* some actions take a while, however we need to ensure that the user instantly knows, that the programm is doing something. This can be ensured by:
   * a message
   * a progressbar indicating that something is happening an how long the action will take approximately. This progessbar also has the advantage of distracting the user, lowering the "subjective time" something takes
* TraceGUI actions like tracing, focussing etc. need to run in background threads, so the main UI thread stays responsive

