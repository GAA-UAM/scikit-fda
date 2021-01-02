{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   
   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}
   {% endblock %}
   
.. include:: {{package}}/backreferences/{{fullname}}.examples