#!/bin/bash
# Open a notebook, first run on geir/yngve, then on lanermac

# Set the ports
REMOTE_PORT=8897
LOCAL_PORT=8889

# Set the machine names
GEIR_MACHINE=geir.astro.utoronto.ca
YNGVE_MACHINE=yngve.astro.utoronto.ca
LOCAL_MACHINE=lanermac.local

# If we're on the server then open the notebook
if [ $HOSTNAME == geir ] || [ $HOSTNAME == geir.astro.utoronto.ca ] || [ $HOSTNAME == yngve ] || [ $HOSTNAME == yngve.astro.utoronto.ca ]; then
  read -p 'notebook or lab? ' JUPYTER_TYPE
  jupyter $JUPYTER_TYPE --no-browser --port=$REMOTE_PORT
fi

# Open the SSH tunnel
if [ $HOSTNAME == $LOCAL_MACHINE ]; then
  read -p 'server name: [geir/yngve] ' SERVER
  echo 'Tunnel open to '$SERVER' through local port '$LOCAL_PORT
  ssh -N -L localhost:$LOCAL_PORT:localhost:$REMOTE_PORT $SERVER
fi
