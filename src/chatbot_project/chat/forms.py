from django import forms

class MessageForm(forms.Form):
    message = forms.CharField(label='Your Message', max_length=1000)
