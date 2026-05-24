Mesh Properties
===============

Surface Area
------------

Per-triangle areas and total surface area:

.. code-block:: python

   from stl import mesh

   m = mesh.Mesh.from_file('model.stl')
   print('Per-triangle areas:', m.areas)
   print('Total surface area:', m.areas.sum())

Bounding Box
------------

.. code-block:: python

   from stl import mesh

   m = mesh.Mesh.from_file('model.stl')
   print('Min corner:', m.min_)
   print('Max corner:', m.max_)

Normals and Unit Normals
------------------------

.. code-block:: python

   from stl import mesh

   m = mesh.Mesh.from_file('model.stl')
   print('Face normals:', m.normals)
   print('Unit normals:', m.units)

Mass Properties
---------------

Compute volume, center of gravity, and inertia tensor.
Requires a closed (watertight) mesh:

.. code-block:: python

   from stl import mesh

   m = mesh.Mesh.from_file('closed_model.stl')
   volume, cog, inertia = m.get_mass_properties()
   print(f'Volume: {volume}')
   print(f'Center of gravity: {cog}')
   print(f'Inertia tensor:\n{inertia}')

.. warning::
   ``get_mass_properties()`` calls ``check(exact=True)``
   internally. If the mesh is not closed, a ``RuntimeError``
   is raised. Use ``is_closed()`` to check beforehand.

Convexity
---------

.. code-block:: python

   from stl import mesh

   m = mesh.Mesh.from_file('model.stl')
   print('Is convex:', m.is_convex())

Cache Invalidation
------------------

Properties like ``areas``, ``centroids``, ``min_``, ``max_``,
and ``units`` are lazily computed and cached. If you modify
vertices after accessing a property, call the corresponding
``update_*`` method to refresh:

.. code-block:: python

   from stl import mesh

   m = mesh.Mesh.from_file('model.stl')
   print(m.areas)  # computed and cached

   m.x += 10  # modify vertices
   m.update_areas()  # refresh cache
   print(m.areas)  # recomputed
