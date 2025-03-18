from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # This must match the redirect in myproject/urls.py
    path('predict/', views.predict, name='predict'),
]

