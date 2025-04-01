Deletion Experiments
===========================================

.. autofunction:: fac.deletion.experiment
.. autofunction:: fac.deletion.experiments

.. autoclass:: fac.deletion.DeletionAccuracyDeltaResult
    :members:

.. autofunction:: fac.deletion.plot_by_deletion_loc_and_affected_site
.. autofunction:: fac.deletion.plot_exon_effects_by_orf
.. autofunction:: fac.deletion.plot_matrix_at_site

Adjacent Deletions Experiment
------------------------------------------

.. autofunction:: fac.deletion.adjacent_coding_exons
.. autofunction:: fac.deletion.close_consecutive_coding_exons
.. py:data:: fac.deletion.conditions

   Each condition reflects the number of deletions being made to the first exon and the second exon.
   Each of these conditions is run when evaluting the model, and indices into this list can be used
   to select the condition to use.

.. autofunction:: fac.deletion.run_on_all_adjacent_deletions
.. autofunction:: fac.deletion.run_on_all_adjacent_deletions_for_multiple_series
.. autofunction:: fac.deletion.plot_adjacent_deletion_results
