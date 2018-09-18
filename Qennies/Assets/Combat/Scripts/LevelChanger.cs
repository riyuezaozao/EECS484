﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class LevelChanger : MonoBehaviour {

    public Animator changer;
    private int leveltoload;
	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
        if (Input.GetMouseButton(0))
        {
            FadeToLevel(0);
        }
        Debug.Log(Input.GetMouseButton(0));
    }
    public void FadeToLevel(int level)
    {
        leveltoload = level;
        changer.SetTrigger("NewStage");
    }
    public void OnFadeComplete()
    {
        SceneManager.LoadScene(leveltoload);
    }

}
